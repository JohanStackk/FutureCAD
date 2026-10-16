# Reinforcement learning

We use veRL with GSPO. The training launcher is
[`scripts/train_rl.sh`](../scripts/train_rl.sh), and the reward is defined in
[`cad_reward.py`](https://github.com/JohanStackk/FutureCAD/blob/main/src/futurecad/rewards/cad_reward.py).
For veRL installation and training infrastructure, see the
[official veRL repository](https://github.com/verl-project/verl). The launcher
uses the veRL 0.6.1 configuration interface.

| Setting | Value |
|---|---|
| Policy loss | GSPO |
| Advantage estimator | Group-normalized advantages (`grpo` in veRL) |
| KL reward / KL loss | Disabled; both coefficients are 0 |
| Clipping offsets | Lower 0.0003, upper 0.0004 |
| Importance-ratio interval | [0.9997, 1.0004] |
| Sampling temperature / top-p / top-k | 1.0 / 1.0 / -1 |
| Responses per prompt | 8 |
| Prompt batch size | 128 |
| PPO mini-batch size | 32 |
| Micro-batch size per GPU | 2 |
| Learning rate | 0.000001 |
| Weight decay | 0.1 |
| Maximum prompt / response tokens | 512 / 4096 |
| Epoch budget | 10 |
| Default hardware | 1 node, 8 GPUs |

## Concurrent CAD rewards

The upstream veRL 0.6.1
[`NaiveRewardManager`](https://github.com/verl-project/verl/blob/v0.6.1/verl/workers/reward_manager/naive.py)
calls the reward function serially for each response. CAD execution and OCC
geometry processing can therefore dominate reward time. We strongly recommend
concurrent reward execution. Our training checkout replaced this serial loop
with a bounded process pool: 64 workers, with a 30-second timeout per response.

The complete replacement is
[`integrations/verl/naive.py`](../integrations/verl/naive.py). In the veRL 0.6.1
source checkout used by the training interpreter, replace
`verl/workers/reward_manager/naive.py` with this file:

```bash
python -m pip install pebble==5.1.3
cp /absolute/path/to/FutureCAD/integrations/verl/naive.py /path/to/verl-0.6.1/verl/workers/reward_manager/naive.py
```

The replacement dispatches individual CAD reward calls to separate worker processes
and writes their scores back in the original response order. Timed-out or
failed jobs receive zero reward. Set `FUTURECAD_REWARD_WORKERS` and
`FUTURECAD_REWARD_TIMEOUT` to adjust concurrency and timeout to the available
CPU, memory, and BRepGround serving capacity. The defaults reproduce the
64-worker, 30-second setup; limit other numerical libraries' CPU threads when
running many workers.

`reward_model.launch_reward_fn_async` moves a whole batch into a Ray worker;
it does not by itself parallelize the serial per-response loop. The original
run left that option disabled and used the process-pool reward manager.
The replacement keeps the reward function and scoring formula unchanged.

With the replacement installed and BRepGround running, launch training from the repository root:

```bash
MODEL_PATH=/path/to/sft-model \
TRAIN_PARQUET=/path/to/your-training.parquet \
bash scripts/train_rl.sh
```

`MODEL_PATH` is the initial SFT policy. `TRAIN_PARQUET` is user-supplied veRL data
with `prompt` chat messages and reward ground truth containing `id` and
`ply_path`. The reward worker needs access to those reference PLY files.
Validation is disabled in the supplied settings. `VAL_PARQUET` defaults to
`TRAIN_PARQUET`; provide separate held-out data before enabling validation.
Extra veRL overrides can be appended to the command, for example
`trainer.total_epochs=1`. Set `RL_PYTHON` if veRL uses a different interpreter.

## RL prompt template

[`prompts/rl_prompt.txt`](../prompts/rl_prompt.txt) is the exact no-think template
used for RL. Replace `{description}` with the CAD model's detailed description.
This is the same template used by `futurecad.prompts.prompt_sft`; the actual RL
data preparation called that function. The template requests a Python code
block defining the final model as `model`.
