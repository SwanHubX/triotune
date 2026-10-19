"""Canonical SFT recipe entry point."""

import asyncio

from triotune.recipes.sft_huanhuan_trio import DEFAULT_CONFIG, load_config, main


if __name__ == "__main__":
    asyncio.run(main())
