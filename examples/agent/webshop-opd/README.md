# WebShop OPD

This example owns its environment adapter in `webshop_env/environment.py`,
data preparation in `webshop_env/data_prepare.py`, and simulator requirements
in `requirements-webshop.txt`. Evaluation and later OPD experiments use these
local modules.

## Install the environment

Use Python 3.12 and Java 11. From the repository root:

```bash
uv sync
uv pip install 'torch>=2.3' --index-url https://download.pytorch.org/whl/cpu
uv pip install -r examples/agent/webshop-opd/requirements-webshop.txt
```

Use the actual path of your Java 11 installation for `JAVA_HOME`; a distribution's
example path may not exist on your machine. Alternatively, place the JDK or a
symlink to it at `data/webshop-opd/java`. The launchers honor a valid `JAVA_HOME`
and fall back to that local installation if the inherited path is missing or
invalid. They set `JVM_PATH` to the selected JDK's `lib/server/libjvm.so`, avoiding
stale pyjnius library overrides. If no valid JDK is found, they stop with a setup
message before opening the simulator. `--check-config` requires no Java runtime.

The simulator imports PyTorch even in text-only mode; CPU PyTorch is sufficient
because model inference runs through PyTRIO. The requirements also install the
pinned spaCy English model. If a later `uv sync` removes these additional
packages, repeat the two `uv pip install` commands before preparing or evaluating.

## Prepare the data

All downloaded WebShop assets, the upstream simulator checkout, and the search
index live under the repository's ignored `data/webshop-opd/` directory:

| Path | Contents |
| --- | --- |
| `data/webshop-opd/assets/` | Pinned product and instruction JSON files |
| `data/webshop-opd/WebShop/` | Pinned upstream simulator checkout |
| `data/webshop-opd/WebShop/data/` | Links to this example's local JSON files |
| `data/webshop-opd/WebShop/search_engine/indexes_1k/` | Generated search index |

Clone the simulator and prepare the assets and index:

```bash
mkdir -p data/webshop-opd
git clone https://github.com/princeton-nlp/WebShop.git data/webshop-opd/WebShop
git -C data/webshop-opd/WebShop checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
bash examples/agent/webshop-opd/eval/scripts/run-student-eval.sh --prepare-data --download
```

The preparation command uses the student's data/environment settings, which
match the teacher's. It makes no model calls. Downloaded files go directly to
`data.directory`; evaluation only reads these local files. An existing index
is reused only when the recorded source and asset hashes match. Conflicting
files or indexes are rejected without overwriting them. Do not add generated
data or the simulator checkout to Git.

## Evaluate before training

See [the evaluation instructions](eval/README.md) for student/teacher launchers,
budgets, output files, and Pass@1/Pass@4 definitions. Both YAML files resolve
data paths relative to their own locations, so launchers work from any directory.

## Train multi-turn vanilla OPD

The repository Triotune package includes a synchronous AgentRL OPD adapter.
The launchers use that shared implementation. See
[the training documentation](train/README.md) for its exact turn-level loss
boundary, offline verification command, configuration check, and later remote
smoke-test command. A separate [TCOD-F2B training entry point](train/README.md#tcod-f2b-with-the-grpo-teacher)
adds an expanding rollout-prefix curriculum using the same OPD interfaces,
using the shared Triotune package and the example-owned WebShop environment.
The training documentation also includes a TCOD-B2F workflow with resumable
Pass@10 expert collection from the GRPO-trained Qwen3.5-4B teacher and
gradient-free environment-prefix replay into a fresh Qwen3.5-4B student.
