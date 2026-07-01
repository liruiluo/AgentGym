import json
from typing import Any, Mapping

import requests

from agentenv.controller import (
    BaseAdapter,
    BaseEnvClient,
    BaseTask,
    extract_python_code_blocks,
    format_code_as_action_prompt,
    format_function_call_prompt,
    parse_python_code_comments,
)
from agentenv.controller.types import (
    ActionFormat,
    ActionWithTought,
    ConversationMessage,
    StepOutput,
)

AGENTMEMORY_FUNCTION_DESCRIPTION = [
    {
        "name": "add",
        "description": "Store a high-value fact into long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key."},
                "value": {"type": "string", "description": "Memory value."},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "update",
        "description": "Update an existing long-term memory by memory_id or key.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory id such as mem_0000."},
                "key": {"type": "string", "description": "Optional memory key."},
                "value": {"type": "string", "description": "New memory value."},
            },
            "required": ["value"],
        },
    },
    {
        "name": "delete",
        "description": "Delete an obsolete or wrong long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory id such as mem_0000."},
                "key": {"type": "string", "description": "Optional memory key."},
            },
        },
    },
    {
        "name": "retrieve",
        "description": "Retrieve relevant long-term memories into active short-term context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Retrieval query."},
                "top_k": {"type": "integer", "description": "Number of memories to retrieve."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "summary",
        "description": "Replace active short-term context with a concise summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Summary text."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "filter",
        "description": "Keep only active short-term context related to the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Filter query."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "buy",
        "description": "Buy a candidate product in the current shopping session.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Visible candidate product id."},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "answer",
        "description": "Provide a final answer or bundle summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Answer text."},
            },
            "required": ["text"],
        },
    },
]

FUNCTION_TO_ACTION = {
    "add": "ADD",
    "update": "UPDATE",
    "delete": "DELETE",
    "retrieve": "RETRIEVE",
    "summary": "SUMMARY",
    "filter": "FILTER",
    "buy": "BUY",
    "answer": "ANSWER",
}


class AgentMemoryAdapter(BaseAdapter):
    conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym, a multi-session memory-dependent agent environment. Every round you receive the current active context and task view. Long-term memory is hidden unless you call RETRIEVE. Use memory tools only when useful, because they have a small cost. Reply in exactly this format:\n\nThought:\nbrief reasoning\n\nAction:\n<one action>\n\nValid actions are ADD/UPDATE/DELETE/RETRIEVE/SUMMARY/FILTER/BUY/ANSWER with a JSON object payload, e.g. ADD {\"key\": \"tv_size\", \"value\": \"The purchased TV is 75 inches.\"} or BUY {\"product_id\": \"mount_b\"}.",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.FUNCTION_CALLING: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym. Long-term memory is hidden unless you retrieve it. Use the available functions to manage memory or perform task actions.\n\n"
                    + format_function_call_prompt(AGENTMEMORY_FUNCTION_DESCRIPTION),
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.CODE_AS_ACTION: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": "You are operating in AgentMemoryGym. Long-term memory is hidden unless you retrieve it. Write Python code to call exactly one available function.\n\n"
                    + format_code_as_action_prompt(AGENTMEMORY_FUNCTION_DESCRIPTION),
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        fn_call = json.loads("{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False)
        thought = fn_call["thought"]
        function_name = fn_call["function_name"]
        arguments = fn_call["arguments"]
        action_name = FUNCTION_TO_ACTION.get(str(function_name).lower())
        if action_name is None:
            raise ValueError(f"Invalid function name: {function_name}")
        return ActionWithTought(thought=thought, action=format_action(action_name, arguments))

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        action_name, arguments = parse_env_action(action_with_thought.action)
        function_name = action_name.lower()
        return json.dumps(
            {
                "thought": action_with_thought.thought,
                "function_name": function_name,
                "arguments": arguments,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        code = extract_python_code_blocks(text)
        action = eval(code, {}, build_code_action_functions())
        thought = parse_python_code_comments(code)
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        action_name, arguments = parse_env_action(action_with_thought.action)
        function_name = action_name.lower()
        return f"```python\n# {action_with_thought.thought}\n{function_name}(**{repr(arguments)})\n```"


class AgentMemoryEnvClient(BaseEnvClient):
    adapter_cls = AgentMemoryAdapter

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        response = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if response.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {response}")
        created = response.json()
        self.env_id = created["id"]
        self.info = {
            "observation": created["observation"],
            "reward": created["reward"],
            "done": created["done"],
            "env_info": created.get("info", {}),
        }
        self.conversation_start = self.adapter_cls.conversation_start_dict[self.action_format]

    def __len__(self):
        return self.data_len

    def post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        response = requests.post(f"{self.env_server_base}/{path}", json=data, timeout=self.timeout)
        assert response.status_code == 200
        return response.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        try:
            parsed_action = self.adapter_cls.action_parser(action, self.action_format)
        except Exception as exc:
            return StepOutput(
                state=f"Invalid Action: {exc}\n\n{self.observe()}",
                reward=-0.1,
                done=False,
            )
        response = self.post("step", {"action": parsed_action})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
            "env_info": response.get("info", {}),
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self.post("reset", {"data_idx": idx})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
            "env_info": response.get("info", {}),
        }
        return response

    def close(self):
        return self.post("close", {})


class AgentMemoryTask(BaseTask):
    env_client_cls = AgentMemoryEnvClient
    env_name = "AgentMemoryGym"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        *args,
        n_clients: int = 1,
        **kwargs,
    ) -> None:
        del args, kwargs
        super().__init__(client_args=client_args, n_clients=n_clients)


def format_action(action_name: str, arguments: dict[str, Any]) -> str:
    return f"{action_name} {json.dumps(arguments, ensure_ascii=False)}"


def parse_env_action(action: str) -> tuple[str, dict[str, Any]]:
    parts = action.strip().split(" ", 1)
    action_name = parts[0]
    if len(parts) == 1:
        return action_name, {}
    return action_name, json.loads(parts[1])


def build_code_action_functions() -> dict[str, Any]:
    def make_function(action_name: str):
        def code_action(**kwargs):
            return format_action(action_name, kwargs)

        return code_action

    return {function_name: make_function(action_name) for function_name, action_name in FUNCTION_TO_ACTION.items()}
