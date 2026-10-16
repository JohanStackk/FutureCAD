#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the SFT model}"
: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to your veRL training data}"
python="${RL_PYTHON:-python}"
train_files="$("$python" -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "$TRAIN_PARQUET")"
val_files="$("$python" -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "${VAL_PARQUET:-$TRAIN_PARQUET}")"
reward_file="$("$python" -c 'from futurecad.rewards import cad_reward; print(cad_reward.__file__)')"
export VLLM_USE_V1=1
export FUTURECAD_GROUNDING_URL="${FUTURECAD_GROUNDING_URL:-http://127.0.0.1:5678}"
export FUTURECAD_REWARD_WORKERS="${FUTURECAD_REWARD_WORKERS:-64}"
export FUTURECAD_REWARD_TIMEOUT="${FUTURECAD_REWARD_TIMEOUT:-30}"

exec "$python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  algorithm.kl_ctrl.kl_coef=0.0 \
  custom_reward_function.path="$reward_file" \
  data.train_files="$train_files" \
  data.val_files="$val_files" \
  data.shuffle=true \
  data.prompt_key=prompt \
  data.truncation=error \
  data.filter_overlong_prompts_workers=128 \
  data.filter_overlong_prompts=true \
  data.train_batch_size=128 \
  data.max_prompt_length=512 \
  data.max_response_length=4096 \
  data.return_multi_modal_inputs=false \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.0003 \
  actor_rollout_ref.actor.clip_ratio_high=0.0004 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.entropy_checkpointing=true \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=13824 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=13824 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.max_num_batched_tokens=4608 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  reward_model.reward_manager=naive \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=false \
  +reward_model.reward_kwargs.max_resp_len=4096 \
  trainer.logger='["console"]' \
  trainer.project_name=futurecad \
  trainer.experiment_name=gspo \
  trainer.n_gpus_per_node="${GPUS_PER_NODE:-8}" \
  trainer.nnodes="${NNODES:-1}" \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=50 \
  trainer.total_epochs=10 \
  trainer.default_local_dir="${OUTPUT_DIR:-outputs/rl}" \
  trainer.resume_mode=auto \
  trainer.log_val_generations=2 \
  "$@"
