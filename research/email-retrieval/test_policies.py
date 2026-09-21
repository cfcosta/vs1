import unittest

from policies import disagreement_hybrid, needs_laya, vote


class PoliciesTest(unittest.TestCase):
    def test_weighted_vote_combines_neighbors_and_clamps_negative_scores(self):
        rows = [("bills", 0.9), ("bulk", 0.6), ("bulk", 0.5), ("bills", -0.8)]
        self.assertEqual(vote(rows), "bulk")
        self.assertEqual(vote([("bills", 0.5), ("bulk", 0.5)]), "bills")
        self.assertIsNone(vote([("bills", -0.1)]))
        self.assertIsNone(vote([]))

    def test_hybrid_uses_laya_when_nearest_labels_disagree(self):
        self.assertEqual(
            disagreement_hybrid([("bulk", 0.8), ("bills", 0.7)], "ops"), "ops"
        )
        self.assertEqual(
            disagreement_hybrid([("bulk", 0.8), ("bulk", 0.7)], "ops"), "bulk"
        )
        self.assertEqual(disagreement_hybrid([], "ops"), "ops")

    def test_routing_can_be_decided_before_model_inference(self):
        self.assertFalse(needs_laya([("bulk", 0.8), ("bulk", 0.7)]))
        self.assertTrue(needs_laya([("bulk", 0.8), ("bills", 0.7)]))
        self.assertTrue(needs_laya([("bulk", -0.1), ("bulk", -0.2)]))
        self.assertTrue(needs_laya([]))
