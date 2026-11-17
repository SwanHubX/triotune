# WebShop AgentRL Training

[![Zhihu Tutorial](https://img.shields.io/badge/Zhihu-WebShop_Tutorial-0084FF?style=flat-square&logo=zhihu&logoColor=white)](https://zhuanlan.zhihu.com/p/2105967931184116262)

This example trains Qwen3.5-4B in the WebShop environment. It includes **GRPO teacher training, vanilla OPD, TCOD-F2B, and TCOD-B2F**. PyTRIO provides model training and inference, while this directory contains the environment adapter, data preparation code, and training entry points.

## Environment Setup

The example requires **Python 3.12, Java 11, and valid PyTRIO credentials**. Run the following commands from the repository root:

```bash
uv sync
uv pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu
uv pip install -r examples/agent/webshop-tcod/requirements-webshop.txt
```

Set JAVA_HOME to a Java 11 installation, or place the JDK under data/webshop-opd/java. CPU-only PyTorch is sufficient because model computation runs through PyTRIO.

The launchers also require envsubst, which is provided by the gettext package.

### Configuration Paths

All YAML paths use PROJECT_ROOT instead of multi-level parent paths:

```yaml
data:
  directory: $PROJECT_ROOT/data/webshop-opd/assets

environment:
  source_dir: $PROJECT_ROOT/data/webshop-opd/WebShop
```

Each launcher obtains the repository root with Git, exports PROJECT_ROOT, renders the selected YAML into a temporary configuration, and removes that file after the command finishes. The same rendering applies when a custom configuration is supplied with --config.

## Build the Dataset

**Use the following commands to prepare the dataset.** They download the pinned data files, link them into WebShop, and build a Lucene index for 1,000 products. No model service is used during this process.

```bash
mkdir -p data/webshop-opd
git clone https://github.com/princeton-nlp/WebShop.git data/webshop-opd/WebShop
git -C data/webshop-opd/WebShop checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
bash examples/agent/webshop-tcod/eval/scripts/run-student-eval.sh --prepare-data --download
```

The generated files are stored in the following locations:

| Path | Contents |
| --- | --- |
| data/webshop-opd/assets/ | Product, generated-instruction, and human-instruction JSON files |
| data/webshop-opd/WebShop/data/ | Links to the local JSON files |
| data/webshop-opd/WebShop/search_engine/indexes_1k/ | Search index for 1,000 products |
| data/webshop-opd/WebShop/.triotune-webshop.json | Dataset version and SHA-256 metadata |

The source is the Hugging Face dataset YWZBrandon/webshop-data at revision ce990fff5aee388db2706f07820c578ab68e0453. The default official split uses **goals 0–499 for testing, goals 500–1499 for validation, and goals starting at 1500 for training**. The current configurations evaluate the first 200 validation goals.

### Dataset Format

GRPO, vanilla OPD, and TCOD-F2B do not use static prompt-answer pairs. Each training row contains only a task identifier and a goal index:

```json
{
  "task_id": "webshop-train-1500",
  "goal_index": 1500
}
```

For each episode, the environment dynamically produces the **shopping instruction, product pages, available actions, and task reward** from goal_index. Model responses use the JSON tool-call format:

```json
{
  "tool_calls": [
    {
      "name": "webshop",
      "arguments": {
        "action": "search[waterproof hiking shoes]"
      }
    }
  ]
}
```

TCOD-B2F additionally requires a cache of successful teacher trajectories. Each JSONL record has the following format:

```json
{
  "goal_index": 1500,
  "success": true,
  "actions": [
    "search[waterproof hiking shoes]",
    "click[B000EXAMPLE]",
    "click[buy now]"
  ]
}
```

## Start Training

Run all commands from the repository root. Add --check-config to any training command to validate its configuration without starting WebShop or a model service.

**Before running an experiment with a GRPO-trained teacher, replace run_grpoexample, ckpt_samplerexample, and ckpt_trainexample in the relevant configuration with identifiers produced by your own GRPO run. Replace your-workspace with your SwanLab workspace name.**

### 1. GRPO Teacher

Train the Qwen3.5-4B teacher used by later OPD and TCOD experiments:

```bash
bash examples/agent/webshop-tcod/train/scripts/run-grpo-teacher.sh
```

### 2. Vanilla OPD

Use the original Qwen3.8-27B teacher:

```bash
bash examples/agent/webshop-tcod/train/scripts/run-vanilla-opd.sh
```

Use the GRPO-trained Qwen3.5-4B teacher:

```bash
bash examples/agent/webshop-tcod/train/scripts/run-vanilla-opd-grpo-teacher.sh
```

### 3. TCOD-F2B

Use the original Qwen3.8-27B teacher:

```bash
bash examples/agent/webshop-tcod/train/scripts/run-tcod-f2b-qwen3.8-27b-teacher.sh
```

Use the GRPO-trained Qwen3.5-4B teacher:

```bash
bash examples/agent/webshop-tcod/train/scripts/run-tcod-f2b-grpo-teacher.sh
```

F2B starts with one turn and **adds one turn every four training steps**, up to 15 turns:

```text
k = min(1 + completed_steps // 4, 15)
```

### 4. TCOD-B2F

B2F uses the GRPO-trained Qwen3.5-4B teacher to generate successful trajectories, then trains a fresh Qwen3.5-4B student. The main launcher collects missing trajectories before starting training:

```bash
bash examples/agent/webshop-tcod/train/scripts/run-tcod-b2f-grpo-teacher.sh
```

To collect teacher trajectories without starting student training:

```bash
bash examples/agent/webshop-tcod/train/scripts/collect-tcod-b2f-grpo-teacher.sh
```

The expert cache is stored under data/webshop-opd/tcod-b2f/ and is excluded from Git.

## Evaluation and Verification

Run the before-training evaluations with:

```bash
bash examples/agent/webshop-tcod/eval/scripts/run-student-eval.sh
bash examples/agent/webshop-tcod/eval/scripts/run-teacher-eval.sh
```

The offline test suite does not access model services, Java, or WebShop data:

```bash
bash examples/agent/webshop-tcod/train/scripts/test-agent-opd.sh
```

Training outputs are written under output/webshop-opd/, including episode records, metrics, reports, and checkpoint manifests. See the [evaluation guide](eval/README.md) for detailed evaluation definitions.
