from __future__ import annotations

import unittest

from src.evaluate.claim_recovery import recover_claims


class ClaimRecoveryTests(unittest.TestCase):
    def test_separates_claims_from_nonclaims_and_unresolved(self) -> None:
        claims, non_claims, unresolved = recover_claims(
            [
                {"id": "1", "llm_label": "supported"},
                {"id": "2", "llm_label": "not_supported"},
                {"id": "3", "llm_label": "not_a_claim"},
                {"id": "4", "llm_label": ""},
            ],
            label_col="llm_label",
        )
        self.assertEqual([row["id"] for row in claims], ["1", "2"])
        self.assertTrue(all(row["is_claim"] for row in claims))
        self.assertEqual([row["id"] for row in non_claims], ["3"])
        self.assertFalse(non_claims[0]["is_claim"])
        self.assertEqual([row["id"] for row in unresolved], ["4"])
        self.assertIsNone(unresolved[0]["is_claim"])
