//! The decision head laya bolts onto the encoder: a small post-hoc
//! transformer, a per-marker scorer, and an action head.
//!
//! The head is a PyTorch `nn.TransformerEncoder` (`norm_first=True`,
//! ReLU feed-forward, `4 * d` wide, `d / 64` heads) and is ported here
//! weight-name for weight-name so a checkpoint loads unchanged.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{
    LayerNorm,
    LayerNormConfig,
    Linear,
    VarBuilder,
    layer_norm,
    linear,
    ops::softmax_last_dim,
};

/// PyTorch's `nn.LayerNorm` default epsilon.
const TORCH_LAYER_NORM_EPS: f64 = 1e-5;

/// Hidden width of the action head's single hidden layer.
const ACTION_HIDDEN: usize = 256;

/// Number of scalar features the action head reads next to the pooled
/// `[CLS]` state: top probability, top-1 minus top-2, normalised
/// entropy, and option count.
pub const ACTION_FEATURES: usize = 4;

fn torch_layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    layer_norm(
        size,
        LayerNormConfig {
            eps: TORCH_LAYER_NORM_EPS,
            ..Default::default()
        },
        vb,
    )
}

/// One `nn.TransformerEncoderLayer` in pre-norm form.
#[derive(Debug, Clone)]
pub struct HeadLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl HeadLayer {
    /// Loads `head.layers.<i>.*`.
    pub fn load(vb: VarBuilder, hidden: usize) -> Result<Self> {
        let num_heads = (hidden / 64).max(1);
        let head_dim = hidden / num_heads;
        let attn = vb.pp("self_attn");
        // PyTorch keeps one fused `in_proj`; split it so `q`, `k`, and
        // `v` come out contiguous, and fold the softmax scale into `q`
        // (weight and bias) so attention runs with scale 1.
        let weight = attn.get((3 * hidden, hidden), "in_proj_weight")?;
        let bias = attn.get(3 * hidden, "in_proj_bias")?;
        let scale = (head_dim as f64).powf(-0.5);
        let q = Linear::new(
            (weight.narrow(0, 0, hidden)? * scale)?,
            Some((bias.narrow(0, 0, hidden)? * scale)?),
        );
        let k = Linear::new(
            weight.narrow(0, hidden, hidden)?.contiguous()?,
            Some(bias.narrow(0, hidden, hidden)?.contiguous()?),
        );
        let v = Linear::new(
            weight.narrow(0, 2 * hidden, hidden)?.contiguous()?,
            Some(bias.narrow(0, 2 * hidden, hidden)?.contiguous()?),
        );
        let out_proj = linear(hidden, hidden, attn.pp("out_proj"))?;
        let linear1 = linear(hidden, 4 * hidden, vb.pp("linear1"))?;
        let linear2 = linear(4 * hidden, hidden, vb.pp("linear2"))?;
        let norm1 = torch_layer_norm(hidden, vb.pp("norm1"))?;
        let norm2 = torch_layer_norm(hidden, vb.pp("norm2"))?;
        Ok(Self {
            q,
            k,
            v,
            out_proj,
            linear1,
            linear2,
            norm1,
            norm2,
            num_heads,
            head_dim,
        })
    }

    /// `x + attn(norm1(x))`, then `x + ffn(norm2(x))`.
    ///
    /// `key_bias` is `(batch, 1, 1, seq)` and additive: `0` on real
    /// tokens and a large negative number on padding, so padded keys
    /// drop out of every softmax while padded queries still get a
    /// finite (and discarded) row.
    pub fn forward(&self, xs: &Tensor, key_bias: &Tensor) -> Result<Tensor> {
        let (b, l, d) = xs.dims3()?;
        let h = xs.apply(&self.norm1)?;
        let heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, l, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = heads(h.apply(&self.q)?)?;
        let k = heads(h.apply(&self.k)?)?;
        let v = heads(h.apply(&self.v)?)?;
        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att = softmax_last_dim(&att.broadcast_add(key_bias)?)?;
        let ctx = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, d))?
            .apply(&self.out_proj)?;
        let xs = (xs + ctx)?;
        self.ffn(&xs)
    }

    /// The same layer over packed `(total_tokens, hidden)` states,
    /// sequence after sequence with no padding: `seqlens` holds the
    /// cumulative sequence starts (`batch + 1` entries) and
    /// `max_seq_len` the longest sequence. Attention is one varlen
    /// flash kernel, so padded keys never exist and nothing is masked.
    #[cfg(feature = "cuda")]
    pub fn forward_packed(
        &self,
        xs: &Tensor,
        seqlens: &Tensor,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let (total, d) = xs.dims2()?;
        let h = xs.apply(&self.norm1)?;
        let shape = (total, self.num_heads, self.head_dim);
        let linear =
            |layer, relu| crate::bias_act_cuda::linear(&h, layer, relu);
        let q = linear(&self.q, false)?.reshape(shape)?;
        let k = linear(&self.k, false)?.reshape(shape)?;
        let v = linear(&self.v, false)?.reshape(shape)?;
        let orig_dtype = q.dtype();
        let flash_dtype = crate::modernbert::flash_compat_dtype(orig_dtype);
        let ctx = candle_flash_attn::flash_attn_varlen(
            &q.to_dtype(flash_dtype)?,
            &k.to_dtype(flash_dtype)?,
            &v.to_dtype(flash_dtype)?,
            seqlens,
            seqlens,
            max_seq_len,
            max_seq_len,
            1.0,
            false,
        )?
        .to_dtype(orig_dtype)?
        .reshape((total, d))?;
        let ctx = crate::bias_act_cuda::linear(&ctx, &self.out_proj, false)?;
        let xs = (xs + ctx)?;
        let h = xs.apply(&self.norm2)?;
        let ffn = crate::bias_act_cuda::linear(&h, &self.linear1, true)?;
        let ffn = crate::bias_act_cuda::linear(&ffn, &self.linear2, false)?;
        xs + ffn
    }

    fn ffn(&self, xs: &Tensor) -> Result<Tensor> {
        let ffn = xs
            .apply(&self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        xs + ffn
    }
}

/// `LayerNorm -> Linear -> GELU -> Linear(1)` read at each marker.
#[derive(Debug, Clone)]
pub struct Scorer {
    norm: LayerNorm,
    dense: Linear,
    out: Linear,
}

impl Scorer {
    /// Loads `scorer.0`, `scorer.1`, `scorer.3`.
    pub fn load(vb: VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            norm: torch_layer_norm(hidden, vb.pp("0"))?,
            dense: linear(hidden, hidden, vb.pp("1"))?,
            out: linear(hidden, 1, vb.pp("3"))?,
        })
    }

    /// `(batch, markers, hidden)` in, `(batch, markers)` F32 logits out.
    pub fn forward(&self, marked: &Tensor) -> Result<Tensor> {
        marked
            .apply(&self.norm)?
            .apply(&self.dense)?
            .gelu_erf()?
            .apply(&self.out)?
            .squeeze(D::Minus1)?
            .to_dtype(DType::F32)
    }
}

/// `Linear(d + 4, 256) -> GELU -> Linear(actions)` over the pooled
/// `[CLS]` state and four confidence features.
#[derive(Debug, Clone)]
pub struct ActionHead {
    dense: Linear,
    out: Linear,
}

impl ActionHead {
    /// Loads `act_head.0` and `act_head.2`. Always runs in F32: it is
    /// tiny and its inputs are already F32.
    pub fn load(vb: VarBuilder, hidden: usize, actions: usize) -> Result<Self> {
        let vb = vb.set_dtype(DType::F32);
        Ok(Self {
            dense: linear(hidden + ACTION_FEATURES, ACTION_HIDDEN, vb.pp("0"))?,
            out: linear(ACTION_HIDDEN, actions, vb.pp("2"))?,
        })
    }

    /// `(batch, hidden)` pooled state and `(batch, 4)` features in,
    /// `(batch, actions)` probabilities out.
    pub fn forward(
        &self,
        pooled: &Tensor,
        features: &Tensor,
    ) -> Result<Tensor> {
        let xs = Tensor::cat(&[pooled, features], 1)?;
        let logits = xs.apply(&self.dense)?.gelu_erf()?.apply(&self.out)?;
        softmax_last_dim(&logits)
    }
}

/// Confidence features for the action head, computed on the host the
/// same way laya does inside its forward pass: from a softmax over the
/// *raw* marker logits (no temperature), with unused marker slots
/// filled with `-1e4`.
pub fn action_features(
    logits: &[f32],
    option_count: usize,
) -> [f32; ACTION_FEATURES] {
    let mut masked: Vec<f32> = logits.to_vec();
    for (i, value) in masked.iter_mut().enumerate() {
        if i >= option_count {
            *value = -1e4;
        }
    }
    let p = softmax(&masked);
    let k = option_count.max(2) as f32;
    let entropy =
        -p.iter().map(|&v| v * v.max(1e-9).ln()).sum::<f32>() / k.ln();
    let mut sorted = p.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let top1 = sorted.first().copied().unwrap_or(0.0);
    let top2 = sorted.get(1).copied().unwrap_or(0.0);
    [top1, top1 - top2, entropy, k / 255.0]
}

/// Plain softmax in F32.
pub fn softmax(z: &[f32]) -> Vec<f32> {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// `1 - H(p) / log(k)`, clipped to `[0, 1]`; `1.0` for fewer than two
/// options.
pub fn confidence_from_probs(p: &[f32], k: usize) -> f32 {
    if k < 2 {
        return 1.0;
    }
    let entropy = -p[..k.min(p.len())]
        .iter()
        .map(|&v| v * v.clamp(1e-12, 1.0).ln())
        .sum::<f32>();
    (1.0 - entropy / (k as f32).ln()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use candle_nn::VarMap;

    use super::*;

    #[test]
    fn softmax_sums_to_one_and_orders_like_input() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn confidence_is_one_when_peaked_and_zero_when_uniform() {
        assert!(
            (confidence_from_probs(&[1.0, 0.0, 0.0], 3) - 1.0).abs() < 1e-6
        );
        assert!(confidence_from_probs(&[1.0 / 3.0; 3], 3).abs() < 1e-6);
        assert_eq!(confidence_from_probs(&[1.0], 1), 1.0);
    }

    #[test]
    fn action_features_ignore_padded_marker_slots() {
        let f = action_features(&[2.0, 0.0, 5.0, 5.0], 2);
        let p = softmax(&[2.0, 0.0]);
        assert!((f[0] - p[0]).abs() < 1e-6);
        assert!((f[1] - (p[0] - p[1])).abs() < 1e-6);
        assert!((f[3] - 2.0 / 255.0).abs() < 1e-7);
        assert!(f[2] > 0.0 && f[2] < 1.0);
    }

    #[test]
    fn head_layer_masks_padded_keys() {
        // Randomly initialised layer: a padded key must not change
        // the output for the real tokens, so a batch of [3 real, 1
        // pad] and the same 3 tokens alone agree on the first three
        // rows.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hidden = 64;
        let layer = HeadLayer::load(vb, hidden).unwrap();
        let full = Tensor::randn(0f32, 1.0, (1, 4, hidden), &device).unwrap();
        let short = full.narrow(1, 0, 3).unwrap();
        let bias_full =
            Tensor::new(&[[[[0f32, 0.0, 0.0, f32::MIN]]]], &device).unwrap();
        let bias_short = Tensor::new(&[[[[0f32, 0.0, 0.0]]]], &device).unwrap();
        let out_full = layer.forward(&full, &bias_full).unwrap();
        let out_short = layer.forward(&short, &bias_short).unwrap();
        let diff = (out_full.narrow(1, 0, 3).unwrap() - out_short)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "padded key leaked into real rows: {diff}");
    }
}
