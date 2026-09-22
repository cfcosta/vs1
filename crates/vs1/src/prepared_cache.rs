//! Bounded exact cache of complete prepared batches, local to one model.
use std::collections::VecDeque;

use super::{EncodedItem, ItemOutput};

const MAX_BYTES: usize = 4 * 1024 * 1024;
struct Entry {
    items: Vec<EncodedItem>,
    outputs: Vec<ItemOutput>,
    modes: [bool; 3],
    bytes: usize,
}
pub(super) struct PreparedCache {
    entries: VecDeque<Entry>,
    capacity: usize,
    bytes: usize,
    #[cfg(test)]
    pub(super) hits: usize,
}
impl PreparedCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            bytes: 0,
            #[cfg(test)]
            hits: 0,
        }
    }
    pub(super) fn get(
        &mut self,
        items: &[&EncodedItem],
        modes: [bool; 3],
    ) -> Option<Vec<ItemOutput>> {
        let index = self.entries.iter().position(|e| {
            e.modes == modes
                && e.items.len() == items.len()
                && e.items.iter().zip(items).all(|(a, b)| a == *b)
        })?;
        #[cfg(test)]
        {
            self.hits += 1;
        }
        let entry = self.entries.remove(index).unwrap();
        let output = entry.outputs.clone();
        self.entries.push_back(entry);
        Some(output)
    }
    pub(super) fn insert(
        &mut self,
        items: &[&EncodedItem],
        outputs: &[ItemOutput],
        modes: [bool; 3],
    ) {
        let bytes = std::mem::size_of::<Entry>()
            + items
                .iter()
                .map(|i| {
                    std::mem::size_of::<EncodedItem>()
                        + i.ids.len() * 4
                        + i.markers.len() * std::mem::size_of::<usize>()
                })
                .sum::<usize>()
            + outputs
                .iter()
                .map(|o| std::mem::size_of::<ItemOutput>() + o.logits.len() * 4)
                .sum::<usize>();
        if self.capacity == 0 || bytes > MAX_BYTES {
            return;
        }
        // Another caller may have filled the same key while inference ran.
        if self.get(items, modes).is_some() {
            return;
        }
        while self.entries.len() >= self.capacity
            || self.bytes + bytes > MAX_BYTES
        {
            self.bytes -= self.entries.pop_front().unwrap().bytes;
        }
        self.entries.push_back(Entry {
            items: items.iter().map(|i| (*i).clone()).collect(),
            outputs: outputs.to_vec(),
            modes,
            bytes,
        });
        self.bytes += bytes;
    }
    #[cfg(test)]
    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
        self.hits = 0;
    }
}
pub(super) fn precision_modes() -> [bool; 3] {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::{
            gemm_reduced_precision_bf16,
            gemm_reduced_precision_f16,
            gemm_reduced_precision_f32,
        };
        [
            gemm_reduced_precision_f32(),
            gemm_reduced_precision_f16(),
            gemm_reduced_precision_bf16(),
        ]
    }
    #[cfg(not(feature = "cuda"))]
    {
        [false; 3]
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuestionKind;
    fn item(id: u32) -> EncodedItem {
        EncodedItem {
            ids: vec![id, 7],
            markers: vec![1],
            kind: QuestionKind::Noul,
        }
    }
    fn output() -> Vec<ItemOutput> {
        vec![ItemOutput {
            logits: vec![0.125],
            act_probability: 0.75,
        }]
    }
    #[test]
    fn key_includes_order_markers_kind_and_precision() {
        let a = item(1);
        let b = item(2);
        let modes = [false; 3];
        let mut cache = PreparedCache::new(4);
        cache.insert(&[&a, &b], &output(), modes);
        assert!(cache.get(&[&a, &b], modes).is_some());
        assert!(cache.get(&[&b, &a], modes).is_none());
        assert!(cache.get(&[&a], modes).is_none());
        assert!(cache.get(&[&a, &b], [true, false, false]).is_none());
        let mut c = a.clone();
        c.markers[0] = 0;
        assert!(cache.get(&[&c, &b], modes).is_none());
        c = a.clone();
        c.kind = QuestionKind::Score;
        assert!(cache.get(&[&c, &b], modes).is_none());
    }
    #[test]
    fn evicts_least_recent_and_rejects_oversized_entries() {
        let (a, b, c) = (item(1), item(2), item(3));
        let modes = [false; 3];
        let mut cache = PreparedCache::new(2);
        cache.insert(&[&a], &output(), modes);
        cache.insert(&[&b], &output(), modes);
        assert!(cache.get(&[&a], modes).is_some());
        cache.insert(&[&c], &output(), modes);
        assert!(cache.get(&[&b], modes).is_none());
        let mut big = item(4);
        big.ids = vec![0; MAX_BYTES / 4];
        cache.insert(&[&big], &output(), modes);
        assert!(cache.get(&[&big], modes).is_none());
        assert!(cache.get(&[&a], modes).is_some());
        assert!(cache.bytes <= MAX_BYTES);
        cache.clear();
        assert_eq!(cache.bytes, 0);
        assert!(cache.get(&[&a], modes).is_none());
    }
}
