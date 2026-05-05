# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class SciWorldWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *ScienceWorldEnv* instance.
    """

    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        import random

        # Add ScienceWorld to path
        sciworld_path = os.path.join(os.path.dirname(__file__), 'ScienceWorld')
        sys.path.insert(0, sciworld_path)

        from scienceworld import ScienceWorldEnv

        # Initialize environment
        jar_path = env_kwargs.get('jar_path')
        env_step_limit = env_kwargs.get('env_step_limit', 100)
        simplifications_preset = env_kwargs.get('simplifications_preset', 'easy')

        self.env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
        self.taskNames = self.env.get_task_names()

        # Handle variations_idx: can be either a flat list or a dict with 'train'/'test' keys
        variations_idx = env_kwargs.get('variations_idx', [])
        if isinstance(variations_idx, dict):
            # If it's a dictionary, flatten all variations from both train and test splits
            self.variations_idx = variations_idx.get('train', []) + variations_idx.get('test', [])
        else:
            # If it's already a flat list, use it directly
            self.variations_idx = variations_idx

        self.simplifications_preset = simplifications_preset

        random.seed(seed)
        self.rng = random.Random(seed)

    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely

        # Get valid actions and objects
        valid_actions = self.env.get_possible_actions()
        valid_objs = self.env.get_possible_objects()
        valid_action_strs = (
            f"Valid_actions: {valid_actions}, "
            f"OBJ needs to be replaced with one of the following objects: {valid_objs}\n"
            f"example: <action>focus on door</action>"
        )

        info['available_actions'] = valid_action_strs
        info['observation_text'] = obs
        info["possible_actions"] = self.env.get_valid_action_object_combinations()

        # 获取当前分数
        current_score = info.get('score', 0.0)
        info['score'] = current_score
        info['task_score'] = current_score

        # 检测子目标完成
        if not hasattr(self, 'prev_score'):
            self.prev_score = 0.0

        # 子目标完成检测: 分数增加表示完成了子目标
        subgoal_completed = (current_score > self.prev_score)

        # 计算奖励(子目标和任务完成是独立的,可以叠加)
        reward = 0.0

        # 检查子目标完成(独立判断)
        if subgoal_completed:
            reward += 1.0
            info['subgoal_completed'] = True
        else:
            info['subgoal_completed'] = False

        # 检查任务完成(独立判断)
        if done and current_score > 0:
            reward += 10.0
            info['won'] = True
        else:
            info['won'] = False

        # 更新prev_score
        self.prev_score = current_score

        return obs, reward, done, info

    def reset(self, variation_idx):
        """Reset the environment with given variation index"""
        # Select task variation
        if variation_idx is None or variation_idx >= len(self.variations_idx):
            # Random selection from available variations
            task_id, task_variation = self.rng.choice(self.variations_idx)
        else:
            # Use specified variation index
            task_id, task_variation = self.variations_idx[variation_idx]

        taskName = self.taskNames[task_id]

        # Load task with simplifications
        simplification_str = self.simplifications_preset if self.simplifications_preset else ""
        self.env.load(taskName, task_variation, simplification_str)
        obs, info = self.env.reset()
        info = dict(info or {})

        # Get task description and valid actions
        task_description = self.env.get_task_description()
        info['task_description'] = task_description

        valid_actions = self.env.get_possible_actions()
        valid_objs = self.env.get_possible_objects()
        valid_action_strs = (
            f"Valid_actions: {valid_actions}, "
            f"OBJ needs to be replaced with one of the following objects: {valid_objs}\n"
            f"example: <action>focus on door</action>"
        )

        info['available_actions'] = valid_action_strs
        info['observation_text'] = obs
        info["possible_actions"] = self.env.get_valid_action_object_combinations()
        info['won'] = False
        info['task_num'] = task_id

        # 初始化prev_score用于子目标检测
        self.prev_score = 0.0
        info['score'] = info.get('score', 0.0)
        info['task_score'] = info['score']
        info['subgoal_completed'] = False

        return obs, info

    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class SciWorldMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *ScienceWorldEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)
        self._env_kwargs = env_kwargs if env_kwargs is not None else {}

        # Set up variation indices for train/val split BEFORE creating workers
        variations_idx = self._env_kwargs.get('variations_idx', {})

        # Extract the appropriate split from the dictionary
        if isinstance(variations_idx, dict):
            # If variations_idx is a dict with 'train'/'test' keys, extract the appropriate split
            if not self.is_train:
                variations_list = variations_idx.get('test', [])
            else:
                variations_list = variations_idx.get('train', [])
        else:
            # Fallback: if it's already a list, use it directly
            variations_list = variations_idx

        # Store the flat list in env_kwargs for workers to use
        self._env_kwargs['variations_idx'] = variations_list

        # Create index range based on the actual list length
        self.variation_idxs = range(len(variations_list))

        print(f"Loaded {len(variations_list)} variations for {'training' if self.is_train else 'testing'}")
        print(f"Variation index range: {self.variation_idxs}")

        # -------------------------- Ray actors setup --------------------------
        # Workers are created AFTER updating _env_kwargs so they receive the flat list
        env_worker = ray.remote(**resources_per_worker)(SciWorldWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Sample variation indices (like WebShop samples goal indices)
        idx = self._rng.choice(self.variation_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)

        # Wait for all workers to close
        ray.get(close_futures)

        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)

        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_sciworld_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_webshop_envs* so higher‑level code can swap seamlessly."""
    return SciWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
