# WebShop GiGPO

## Install the environment

TrioTune requires Python 3.12. The WebShop simulator also requires Git and
Java 11. Install the project and the WebShop dependencies from the repository
root:

```bash
uv sync --extra ppo
uv pip install -r examples/gigpo/requirements-webshop.txt
uv run python -m spacy download en_core_web_sm
```

Set `JAVA_HOME` to a Java 11 installation. For example, a common Linux path is:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

Clone the pinned Princeton WebShop source revision:

```bash
git clone https://github.com/princeton-nlp/WebShop.git .cache/webshop
git -C .cache/webshop checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
```

## Prepare the data

Download the pinned WebShop assets, link them into the source checkout, and
build the local search index:

```bash
uv run python -m examples.gigpo.data_prepare \
  --config examples/gigpo/configs/gigpo-webshop-configs.yaml \
  --download
```

The command reuses the default Hugging Face cache and writes the search index
under `.cache/webshop/search_engine/indexes_1k`.

## Start training

Run the original GiGPO launcher from any working directory:

```bash
bash examples/gigpo/scripts/run-gigpo-webshop.sh
```

Extra command line arguments are forwarded to the training entry point. For
example, run a local dependency and environment check without starting remote
training:

```bash
bash examples/gigpo/scripts/run-gigpo-webshop.sh --mode check
```
