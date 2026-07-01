from __future__ import annotations

import threading
from typing import Any

from .environment import AgentMemoryEnv, load_task_dataset


class AgentMemoryWrapper:
    def __init__(self) -> None:
        self.max_id = 0
        self.tasks = load_task_dataset()
        self.envs: dict[int, AgentMemoryEnv] = {}
        self.info: dict[int, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create(self) -> dict[str, Any]:
        with self.lock:
            env_id = self.max_id
            self.max_id += 1
        env = AgentMemoryEnv(tasks=self.tasks)
        observation, info = env.reset(data_idx=env_id)
        self.envs[env_id] = env
        self.info[env_id] = {
            "observation": observation,
            "reward": 0.0,
            "done": False,
            "info": info,
        }
        return {"id": env_id, "observation": observation, "reward": 0.0, "done": False, "info": info}

    def step(self, env_id: int, action: str) -> dict[str, Any]:
        env = self.require_env(env_id)
        observation, reward, done, _, info = env.step(action)
        payload = {"observation": observation, "reward": reward, "done": done, "info": info}
        self.info[env_id] = payload
        return payload

    def reset(self, env_id: int, data_idx: int = 0) -> dict[str, Any]:
        env = self.require_env(env_id)
        observation, info = env.reset(data_idx=data_idx)
        payload = {"id": env_id, "observation": observation, "reward": 0.0, "done": False, "info": info}
        self.info[env_id] = payload
        return payload

    def observation(self, env_id: int) -> str:
        self.require_env(env_id)
        return self.info[env_id]["observation"]

    def detail(self, env_id: int) -> dict[str, Any]:
        self.require_env(env_id)
        return self.info[env_id]

    def close(self, env_id: int) -> bool:
        env = self.require_env(env_id)
        env.close()
        del self.envs[env_id]
        del self.info[env_id]
        return True

    def metadata(self) -> dict[str, Any]:
        return {
            "task_count": len(self.tasks),
            "task_ids": [task.task_id for task in self.tasks],
            "splits": sorted({task.split for task in self.tasks}),
            "source": sorted({task.source for task in self.tasks}),
        }

    def require_env(self, env_id: int) -> AgentMemoryEnv:
        if env_id not in self.envs:
            raise KeyError(f"Unknown environment id {env_id}.")
        return self.envs[env_id]


server = AgentMemoryWrapper()
