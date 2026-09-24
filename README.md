# trio-cookbook
用PyTRIO做模型训练案例、RL训练框架设计

## 长期目标

- 专门用于训练大模型RL算法的框架，要求实现通用性、实用性
- 基于框架提供实际案例

## 框架适配

PPO一类，GRPO一类，DPO一类

期望使用类似于TRL的Trainer形式实现基于PyTRIO的强化学习训练

## 环境安装

使用 uv（推荐，依赖版本由 `uv.lock` 锁定）：

```bash
uv sync
uv run python --version
# The local package is installed by uv, so this import works:
uv run python -c "import trio_cookbook"
```

也可以继续使用 requirements 文件：

```bash
pip install -r requirements.txt
# Install this repository as well when using pip directly:
pip install -e .
```
