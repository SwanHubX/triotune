"""数据准备：复用 HF 缓存、构建索引、划分目标并生成任务行。

商品 JSON 不是监督学习的 prompt/answer 数据。这里的任务行只指定目标索引；
真实指令、页面和奖励由 WebShop 在 reset/step 时产生，不创建占位 parquet。
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import importlib
import json
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from huggingface_hub import hf_hub_download

WEBSHOP_REPOSITORY = "https://github.com/princeton-nlp/WebShop.git"
WEBSHOP_REVISION = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"


def asset_manifest(files: dict[str, Path]) -> dict:
    """对 files 的文件名→本地路径映射计算 SHA-256，返回源码版本及资产清单。"""
    return {
        "webshop_repository": WEBSHOP_REPOSITORY,
        "webshop_revision": WEBSHOP_REVISION,
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())
        },
    }


@dataclass(frozen=True)
class DataConfig:
    """repo_id/revision 固定 HF 数据；local_files_only 控制是否只查缓存。

    split_protocol 选择索引协议；seed 仅控制训练行打乱；train_epochs 为任务遍历
    次数；eval_limit 是显式调试子集大小，None 表示完整评估区间。
    """

    repo_id: str = "YWZBrandon/webshop-data"
    revision: str = "ce990fff5aee388db2706f07820c578ab68e0453"
    local_files_only: bool = True
    split_protocol: str = "official"
    seed: int = 42
    train_epochs: int = 2
    eval_limit: int | None = None

    def __post_init__(self):
        """验证划分、随机种子、遍历次数和子集大小，拒绝 bool 伪装成整数。"""
        if self.split_protocol not in {"verl", "official"}:
            raise ValueError("data.split_protocol must be verl or official.")
        for name, minimum in (("seed", 0), ("train_epochs", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"data.{name} must be an integer >= {minimum}.")
        if self.eval_limit is not None and (
            type(self.eval_limit) is not int or self.eval_limit < 1
        ):
            raise ValueError("data.eval_limit must be a positive integer or null.")
        if type(self.local_files_only) is not bool:
            raise ValueError("data.local_files_only must be boolean.")


def cached_files(config: DataConfig) -> dict[str, Path]:
    """不指定 cache_dir/local_dir，直接复用用户之前下载的默认 HF 缓存。

    第三个文件是该版本 WebShop loader 无条件读取的依赖；human_goals=False
    时不会用其中的人工指令生成任务。数据缺失时保留 HF 的明确异常。
    """
    return {
        name: Path(
            hf_hub_download(
                repo_id=config.repo_id,
                repo_type="dataset",
                revision=config.revision,
                filename=name,
                local_files_only=config.local_files_only,
            )
        )
        for name in (
            "items_shuffle_1000.json",
            "items_ins_v2_1000.json",
            "items_human_ins.json",
        )
    }


def goal_ranges(count: int, protocol: str) -> dict[str, range]:
    """以目标总数 count 和划分协议 protocol 返回 train/eval/test 的 range。

    索引指向固定 seed 生成并打乱后的 server.goals，不是商品编号。

    verl 协议显式承认 eval 使用原来的 test 区间；official 协议保留独立 test，
    避免在训练过程中反复用 test 选择 checkpoint。两者都是目标级划分，
    商品仍共享，不能将结果解释为对未见商品的泛化能力。
    """
    if protocol not in {"verl", "official"}:
        raise ValueError("Unknown split protocol.")
    start = 500 if protocol == "verl" else 1500
    if type(count) is not int or count <= start:
        raise ValueError(
            f"Need more than {start} goals for protocol={protocol}; got {count}."
        )
    return {
        "train": range(start, count),
        "eval": range(500) if protocol == "verl" else range(500, 1500),
        "test": range(500),
    }


def task_rows(goals: list[dict], config: DataConfig, split: str) -> list[dict]:
    """将有序 goals 按 DataConfig 和 split 转为可重放任务行列表。

    仅训练行按 epoch 打乱，评估顺序固定；split 为 train/eval/test。

    同一目标的不同候选由 AgentRolloutWorkflow 扩展，不能预先复制成不同组。
    task_id 绑定目标索引和划分，环境状态及奖励答案均不写入 messages。
    """
    indices = list(goal_ranges(len(goals), config.split_protocol)[split])
    if split != "train" and config.eval_limit is not None:
        indices = indices[: config.eval_limit]
    rows = []
    rng = random.Random(config.seed)
    for _ in range(config.train_epochs if split == "train" else 1):
        if split == "train":
            rng.shuffle(indices)
        rows.extend(
            {"task_id": f"webshop-{split}-{i}", "goal_index": i} for i in indices
        )
    return rows


def goal_fingerprint(goals: list[dict]) -> str:
    """记录完整目标顺序和价格约束，便于发现换数据/seed 后的不可比评估。"""
    payload = json.dumps(goals, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def link_assets(source: Path, files: dict[str, Path]):
    """HF 缓存为源，外部源码的 data/ 只建立符号链接；绝不覆盖已有不同文件。"""
    directory = source / "data"
    directory.mkdir(parents=True, exist_ok=True)
    for name, target in files.items():
        destination = directory / name
        if destination.exists() and filecmp.cmp(destination, target, shallow=False):
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace existing WebShop data: {destination}"
            )
        destination.symlink_to(target.resolve())


def prepare(experiment, *, download: bool = False):
    """按 Experiment 定位源码和数据，构建匹配索引，返回资产 SHA-256 清单。

    download=True 允许补齐 HF 缓存；默认遵循 data.local_files_only。任何未知
    索引或不同内容的数据链接都报错，已匹配的准备结果可以重复复用。
    """
    source = experiment.source_dir
    if not (source / "web_agent_site").is_dir():
        raise FileNotFoundError(
            f"Clone the pinned Princeton WebShop checkout first: {source}; see README.md."
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != WEBSHOP_REVISION:
        raise ValueError(
            f"Expected WebShop source revision {WEBSHOP_REVISION}; got {revision}."
        )
    files = cached_files(
        replace(experiment.data, local_files_only=False)
        if download
        else experiment.data
    )
    expected = asset_manifest(files)
    manifest = source / ".triotune-webshop.json"
    index = source / "search_engine/indexes_1k"
    if index.exists():
        if not manifest.is_file() or json.loads(manifest.read_text()) != expected:
            raise FileExistsError(
                "Existing indexes_1k has unknown/different data; use a clean source checkout."
            )
        link_assets(source, files)
        return expected
    link_assets(source, files)
    sys.path.insert(0, str(source))
    engine = importlib.import_module("web_agent_site.engine.engine")
    if not Path(engine.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("Another WebShop checkout is already imported.")
    # 文档内容完全沿用上游 convert_product_file_format.py；只构建实际使用的 1k。
    # 原始 loader 从源码 data/ 读取属性，不接受 attrpath 参数；上面已校验链接。
    random_state = random.getstate()
    try:
        random.seed(experiment.environment_seed)
        products, *_ = engine.load_products(
            filepath=str(files["items_shuffle_1000.json"]),
            num_products=1000,
            human_goals=False,
        )
    finally:
        random.setstate(random_state)
    # 临时目录只在构建成功后改名；中途失败不会留下被误认成完整索引的目录。
    with tempfile.TemporaryDirectory(
        prefix="triotune-index-", dir=index.parent
    ) as temporary:
        temporary = Path(temporary)
        documents = temporary / "documents"
        documents.mkdir()
        with (documents / "documents.jsonl").open("w", encoding="utf-8") as stream:
            for product in products:
                options = ", and ".join(
                    f"{name}: {', '.join(values)}"
                    for name, values in product.get("options", {}).items()
                )
                contents = " ".join(
                    [
                        product["Title"],
                        product["Description"],
                        product["BulletPoints"][0],
                        options,
                    ]
                ).lower()
                stream.write(
                    json.dumps(
                        {
                            "id": product["asin"],
                            "contents": contents,
                            "product": product,
                        }
                    )
                    + "\n"
                )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                str(documents),
                "--index",
                str(temporary / "index"),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                "1",
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ],
            check=True,
        )
        (temporary / "index").rename(index)
    manifest.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return expected


def main(argv=None):
    """解析配置路径及 --download，执行准备并输出 JSON 清单，不连接训练服务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "grpo-webshop-configs.yaml",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing small assets into the default HF cache",
    )
    args = parser.parse_args(argv)
    # 延迟导入编排层，数据工具的普通导入不创建训练服务，也不产生循环依赖。
    from .train import load_experiment

    experiment = load_experiment(args.config)
    result = prepare(experiment, download=args.download)
    print(json.dumps({"source_dir": str(experiment.source_dir), **result}, indent=2))


if __name__ == "__main__":
    main()
