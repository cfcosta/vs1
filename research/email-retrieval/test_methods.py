import unittest

from methods import independent, nearest


class MethodsTest(unittest.TestCase):
    def test_sender_and_template_exclusion(self):
        a = {
            "index": 0,
            "sender": "a",
            "body": "This is a payment receipt number 123 from the shop for your purchase",
        }
        b = dict(a, index=1, sender="b", body=a["body"].replace("123", "456"))
        c = dict(a, index=2, body="Independent content")
        d = dict(
            a, index=3, sender="c", body="An interview invitation unrelated to shopping"
        )
        self.assertEqual(independent([a], [b, c, d]), [d])

    def test_nearest_uses_only_training_candidates(self):
        self.assertEqual(nearest([[1, 0], [0, 1], [0.9, 0.1]], 0, [1, 2]), [2, 1])

    def test_nearest_rejects_self(self):
        with self.assertRaises(ValueError):
            nearest([[1, 0]], 0, [0])


if __name__ == "__main__":
    unittest.main()
