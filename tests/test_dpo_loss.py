"""Failure paths for the extracted local DPO objective."""

import unittest

import torch

from trio_cookbook.trainer.dpo.trainer import _datum as build_datum
from trio_cookbook.trainer.dpo.loss import make_loss_fn


class DPOLossTests(unittest.TestCase):
    def test_pair_and_reference_counts_must_align(self):
        datum = build_datum([10], [11])
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
        datums = [build_datum([10, 11], [12]), build_datum([10, 11], [13])]
        logprobs = [torch.tensor([-0.5], requires_grad=True)] * 2
        with self.assertRaisesRegex(ValueError, "not aligned"):
            make_loss_fn([-0.5, -0.5], 0.1)(datums, logprobs)
