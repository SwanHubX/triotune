from __future__ import annotations

import unittest

from trio_cookbook import GRPOConfig
from trio_cookbook.trainer.common.client import sampling_params


class _SamplingParams:
    def __init__(self, **values: object) -> None:
        self.values = values


class SamplingParamsTests(unittest.TestCase):
    def test_stop_is_added_when_provided(self) -> None:
        config = GRPOConfig()
        params = sampling_params(config, _SamplingParams, stop=["<|im_end|>"])

        self.assertEqual(params.values["stop"], ["<|im_end|>"])

    def test_stop_is_omitted_when_disabled(self) -> None:
        config = GRPOConfig(stop=False)
        params = sampling_params(config, _SamplingParams)

        self.assertNotIn("stop", params.values)


if __name__ == "__main__":
    unittest.main()
