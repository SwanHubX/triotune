"""Failure paths for the extracted local DPO objective."""

import unittest

import torch

from triotune.trainer.dpo.datum import build_dpo_datum
from triotune.trainer.dpo.loss import dpo_evaluation_metrics, make_loss_fn


class DPOLossTests(unittest.TestCase):
    def test_pair_and_reference_counts_must_align(self):
        datum = build_dpo_datum([10], [11])
        logprob = torch.tensor([-0.5], requires_grad=True)
        for datums, logprobs, reference in (
            ([datum], [logprob], [-0.5]),
            ([datum, datum], [logprob], [-0.5, -0.5]),
            ([datum, datum], [logprob, logprob], [-0.5]),
        ):
            with self.subTest(counts=(len(datums), len(logprobs), len(reference))):
                with self.assertRaisesRegex(ValueError, "aligned chosen/rejected"):
                    make_loss_fn(reference, 0.1)(datums, logprobs)

    def test_policy_tokens_must_align_with_shifted_labels(self):
        datums = [
            build_dpo_datum([10, 11], [12]),
            build_dpo_datum([10, 11], [13]),
        ]
        logprobs = [torch.tensor([-0.5], requires_grad=True)] * 2
        with self.assertRaisesRegex(ValueError, "not aligned"):
            make_loss_fn([-0.5, -0.5], 0.1)(datums, logprobs)

    def test_evaluation_metrics_use_pairwise_dpo_objective(self):
        metrics = dpo_evaluation_metrics(
            [-1.0, -2.0],
            [-1.5, -1.5],
            beta=0.1,
        )

        self.assertGreater(metrics["eval/reward_margin"], 0)
        self.assertEqual(metrics["eval/reward_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["eval/logratio_margin"], 1.0)
        self.assertEqual(metrics["eval/reference_logp_chosen"], -1.5)
        self.assertEqual(metrics["eval/pairs"], 1.0)
        with self.assertRaisesRegex(ValueError, "aligned pairs"):
            dpo_evaluation_metrics([-1.0], [-1.0], beta=0.1)
