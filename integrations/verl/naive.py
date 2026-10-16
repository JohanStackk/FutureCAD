# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modified by the FutureCAD authors for concurrent CAD reward execution.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from collections import defaultdict
from concurrent.futures import TimeoutError, as_completed
from typing import Any

import torch
from pebble import ProcessPool
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_workers = int(os.environ.get("FUTURECAD_REWARD_WORKERS", "64"))
        self.timeout_seconds = float(os.environ.get("FUTURECAD_REWARD_TIMEOUT", "30"))
        if self.max_workers < 1 or not 0 < self.timeout_seconds < float("inf"):
            raise ValueError(
                "Reward workers and timeout must be positive finite values"
            )
        if overlong_buffer_cfg and overlong_buffer_cfg.get("enable", False):
            raise ValueError("This reward manager does not enable overlong penalties")

    def __call__(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch:
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prepared = []
        for index in range(len(data)):
            item = data[index]
            prompt_length = item.batch["prompts"].shape[-1]
            response_length = int(
                item.batch["attention_mask"][prompt_length:].sum().item()
            )
            if response_length == 0:
                continue
            if not 0 < response_length <= item.batch["responses"].shape[-1]:
                raise ValueError("Response length disagrees with the attention mask")
            response = self.tokenizer.decode(
                item.batch["responses"][:response_length], skip_special_tokens=True
            )
            extra_info = dict(item.non_tensor_batch.get("extra_info") or {})
            extra_info["num_turns"] = item.non_tensor_batch.get("__num_turns__")
            extra_info["rollout_reward_scores"] = item.non_tensor_batch.get(
                "reward_scores", {}
            )
            prepared.append(
                (
                    index,
                    response_length,
                    {
                        "data_source": item.non_tensor_batch[self.reward_fn_key],
                        "solution_str": response,
                        "ground_truth": item.non_tensor_batch["reward_model"][
                            "ground_truth"
                        ],
                        "extra_info": extra_info,
                    },
                )
            )

        scores = {}
        if prepared:
            with ProcessPool(max_workers=min(self.max_workers, len(prepared))) as pool:
                pending = {
                    pool.schedule(
                        self.compute_score, kwargs=kwargs, timeout=self.timeout_seconds
                    ): (index, length)
                    for index, length, kwargs in prepared
                }
                for future in as_completed(pending):
                    index, length = pending[future]
                    try:
                        score = future.result()
                    except TimeoutError:
                        logger.warning("CAD reward timed out for response %s", index)
                        score = 0.0
                    except Exception:
                        logger.exception("CAD reward failed for response %s", index)
                        score = 0.0
                    scores[index] = (length, score)

        for index in sorted(scores):
            length, score = scores[index]
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            reward_tensor[index, length - 1] = float(reward)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
