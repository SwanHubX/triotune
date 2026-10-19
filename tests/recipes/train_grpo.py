"""Canonical GRPO recipe entry point.

The implementation remains in :mod:`grpo_gms8k_trio` for compatibility with
existing users; this short name is used by the launch scripts.
"""

from triotune.recipes.grpo_gms8k_trio import DEFAULT_CONFIG, load_config, main


if __name__ == "__main__":
    main()
