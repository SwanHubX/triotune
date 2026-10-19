"""Canonical asynchronous OPD recipe entry point."""

import asyncio

from trio_cookbook.recipes.opd_deepmath_trio import DEFAULT_CONFIG, load_config, main


if __name__ == "__main__":
    asyncio.run(main())
