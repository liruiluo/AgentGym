from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    ConversationMessage,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from .filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER,
    FILESYSTEM_CHECKPOINT_MAX_BYTES,
    FILESYSTEM_CHECKPOINT_PATH,
    FILESYSTEM_CHECKPOINT_REQUEST,
    checkpoint_retry_trigger_tokens,
    build_filesystem_checkpoint_read_retry_observation,
    build_filesystem_checkpoint_retry_observation,
    build_post_checkpoint_context,
    build_post_checkpoint_read_retry_context,
    filesystem_checkpoint_failure_reason,
    filesystem_checkpoint_framing_sha256,
    filesystem_checkpoint_read_failure_reason,
    filesystem_checkpoint_read_observed,
    filesystem_checkpoint_write_succeeded,
    normalize_filesystem_checkpoint_receipt,
)


# The route-level forecast must cover the largest policy-visible observation
# seen with the frozen LiteResearcher service and Qwen3.5 tokenizer: the
# maximum r43 next-prompt growth was 10,652 tokens, comprising a 52-token
# response plus a 10,600-token observation-and-template delta. Keep a bounded
# margin so compaction is sampled before the
# next native action can push the prompt past the 30,720-token PPO width.
LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE = 12_288


LITERESEARCHER_CONTEXT_COMPACTION_REQUEST = (
    FILESYSTEM_CHECKPOINT_REQUEST
    + " For this research task, preserve the unresolved question, decisive "
    "evidence with source URLs, conflicts or uncertainty, and the next search "
    "or visit action."
)
LITERESEARCHER_POLICY_CONTINUATION_MARKER = (
    FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER
)



LITERESEARCHER_SYSTEM_PROMPT = """# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "search", "description": "Search the released web corpus with one or more queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit one opaque URL returned by search.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "goal": {"type": "string"}, "page": {"type": "integer", "minimum": 1}}, "required": ["url", "goal"]}}}
</tools>

For a search, use this complete form. The query value MUST be a JSON array of
one or more non-empty strings, never a single string:

<tool_call>
<function=search>
<parameter=query>
["first search query", "second search query"]
</parameter>
</function>
</tool_call>

For a visit, use this complete form. Replace the URL with one copied verbatim
from a search result; never invent, reconstruct, shorten, or edit a URL:

<tool_call>
<function=visit>
<parameter=url>
URL_COPIED_VERBATIM_FROM_A_SEARCH_RESULT
</parameter>
<parameter=goal>
specific evidence to find on that page
</parameter>
<parameter=page>
1
</parameter>
</function>
</tool_call>

Function names are limited to search and visit. Never write
<function=answer>, <function=apply_patch>, or <function=shell_command>.
Do not wrap workspace actions or the final answer in <tool_call> tags.
Required parameters must be present. Emit no text after a function call.

You are a meticulous deep-research agent working on one continuous question. Research before answering. On the first turn, call search even if the answer seems obvious. Copy each visit URL exactly from a search result. A visit returns one bounded page; follow next_page with the same URL and goal when needed.

An empty episode-private workspace persists across context compaction. Use files when evidence or a continuation plan should survive a long interaction.
At an explicit context-boundary request, use one normal shell_command or apply_patch action to overwrite `.agent_memory/CONTINUATION.md`; only a verified non-empty write allows old messages to be removed. After replacement, read that file through a normal action before continuing. Other workspace files remain available for voluntary notes at any time.

Valid shell action:
shell_command {"command":"cat .agent_memory/research.md","workdir":".","timeout_ms":10000}

Valid file edit action:
apply_patch
*** Begin Patch
*** Add File: .agent_memory/research.md
+Question, evidence, source URLs, and next steps.
*** End Patch

When evidence is sufficient, use this complete final form and replace the text
with the evidence-backed answer:
<answer>your evidence-backed answer</answer>

Emit exactly one function call, workspace action, or final answer per turn."""

class LiteResearcherEnvClient(BaseEnvClient):
    """Task-neutral LiteResearcher client with policy-authored compaction."""

    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": LITERESEARCHER_SYSTEM_PROMPT,
            }
        ),
        ConversationMessage(
            {"from": "gpt", "loss": False, "value": "Understood."}
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int | None = None,
        *args,
        timeout: int = 900,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        metadata = self._request("GET", "metadata")
        if metadata.get("domain_id") != "literesearcher":
            raise RuntimeError("LiteResearcher endpoint reports the wrong domain")
        if (
            metadata.get("compaction_contract")
            != "policy_filesystem_checkpoint_then_client_replace_v2"
        ):
            raise RuntimeError("LiteResearcher endpoint reports the wrong compaction contract")
        task_count = int(metadata["task_count"])
        if data_len is not None and int(data_len) > task_count:
            raise ValueError(
                f"LiteResearcher data_len {data_len} exceeds task_count {task_count}"
            )
        self.data_len = task_count if data_len is None else int(data_len)
        created = self._request("POST", "create", json={})
        self.env_id = int(created["id"])
        self.info = created
        self.metadata = metadata
        self._reset_policy_transition_state()

    def _reset_policy_transition_state(self) -> None:
        self._policy_step_count = 0
        self._native_call_count = 0
        self._context_epoch = 0
        self._immutable_policy_context: list[dict[str, str]] | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._policy_context_bound = False
        self._selected_policy_control: str | None = None
        self._checkpoint_retry_pending = False
        self._pending_checkpoint_read: dict[str, Any] | None = None
        self._pending_checkpoint_read_framing: list[dict[str, str]] | None = None

    def __len__(self) -> int:
        return self.data_len

    def observe(self) -> str:
        return str(self.info["observation"])

    @property
    def sample_excluded(self) -> bool:
        return bool(self.info.get("info", {}).get("sample_excluded", False))

    def policy_framing(self) -> list[dict[str, str]]:
        """Expose the exact immutable prompt used by this wrapper."""

        return [{"role": "system", "content": LITERESEARCHER_SYSTEM_PROMPT}]

    def normalize_initial_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        normalized = _copy_policy_messages(messages)
        if not normalized or normalized[-1]["role"] != "user":
            raise ValueError(
                "LiteResearcher initial policy context must end with the question"
            )
        observation = str(self.observe())
        if normalized[-1]["content"] != observation:
            raise ValueError(
                "LiteResearcher initial policy context does not end with the current question"
            )
        return self.policy_framing() + [{"role": "user", "content": observation}]

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        normalized = _copy_policy_messages(messages)
        if initial:
            self._immutable_policy_context = deepcopy(normalized)
            self._policy_context_bound = True
        self._current_policy_context = normalized

    def policy_turn_candidate(self) -> str | None:
        if not self._policy_context_bound or self._pending_checkpoint_read is not None:
            return None
        return LITERESEARCHER_CONTEXT_COMPACTION_REQUEST

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        self._selected_policy_control = None
        if not self._policy_context_bound:
            return None
        if self._pending_checkpoint_read is not None:
            return None
        if pressure is None:
            raise RuntimeError(
                "LiteResearcher compaction requires task-neutral token pressure"
            )
        if (
            pressure.max_observation_tokens
            < LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE
        ):
            raise RuntimeError(
                "LiteResearcher route observation-token envelope is too small: "
                f"configured={pressure.max_observation_tokens} "
                f"minimum={LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE}"
            )
        capacity = pressure.effective_prompt_capacity
        if (
            pressure.action_prompt_tokens > capacity
            or pressure.candidate_prompt_tokens > capacity
        ):
            raise RuntimeError(
                "LiteResearcher context reached the prompt cap before a trainable "
                "compaction could be sampled"
            )
        # Decide from the no-control append path.  Continuous Token chat
        # normalization may make the rendered control candidate shorter than the
        # ordinary action prompt, so candidate-minus-action is not a valid size
        # or safety invariant.
        if (
            not self._checkpoint_retry_pending
            and checkpoint_retry_trigger_tokens(
                pressure, control_request=LITERESEARCHER_CONTEXT_COMPACTION_REQUEST
            )
            < capacity
        ):
            return None
        self._selected_policy_control = "context_compaction"
        return LITERESEARCHER_CONTEXT_COMPACTION_REQUEST

    def step(self, action: str) -> StepOutput:
        if self._selected_policy_control == "context_compaction":
            return self._complete_context_compaction(action)
        return self._step_native_policy_action(action)

    def _step_native_policy_action(self, action: str) -> StepOutput:
        native_before = self._native_call_count
        policy_before = self._policy_step_count
        context_before = self._context_epoch
        checkpoint_read_pending_before = self._pending_checkpoint_read
        checkpoint_read_framing_before = self._pending_checkpoint_read_framing
        response = self._request(
            "POST",
            "step",
            json={"id": self.env_id, "action": action},
        )
        self._native_call_count += 1
        self._policy_step_count += 1
        self.info = response
        response_info = response.get("info", {})
        action_submission = response_info.get("action_submission")
        if not isinstance(action_submission, Mapping):
            action_submission = {"raw_policy_output": action}
        server_wrapper = (
            response_info.get("wrapper_evidence", {})
            if isinstance(response_info, Mapping)
            else {}
        )
        wrapper_evidence: dict[str, Any] = {
            "event": "native_action",
            "server_wrapper_evidence": (
                dict(server_wrapper) if isinstance(server_wrapper, Mapping) else {}
            ),
        }
        read_receipt = None
        if isinstance(server_wrapper, Mapping):
            read_receipt = server_wrapper.get("filesystem_checkpoint_read")
            if filesystem_checkpoint_read_observed(read_receipt):
                wrapper_evidence.update(
                    {
                        "memory_event": "read",
                        "document_read_observed": True,
                        "filesystem_checkpoint_read": dict(read_receipt),
                    }
                )
            else:
                changed_paths = server_wrapper.get("workspace_changed_paths")
                noncheckpoint_paths = (
                    sorted(
                        {
                            str(path)
                            for path in changed_paths
                            if isinstance(path, str)
                            and path
                            and path != FILESYSTEM_CHECKPOINT_PATH
                        }
                    )
                    if isinstance(changed_paths, Sequence)
                    and not isinstance(changed_paths, (str, bytes))
                    else []
                )
                if (
                    server_wrapper.get("workspace_action_completed") is True
                    and noncheckpoint_paths
                ):
                    wrapper_evidence.update(
                        {
                            "memory_event": "modify",
                            "workspace_change_observed": True,
                            "workspace_changed_paths": noncheckpoint_paths,
                        }
                    )
                elif (
                    server_wrapper.get("workspace_action_completed") is True
                    and str(server_wrapper.get("workspace_op", "")).upper()
                    == "SHELL_COMMAND"
                ):
                    wrapper_evidence.update(
                        {
                            "memory_event": "execute",
                            "outcome": "success",
                            "execution_completed_observed": True,
                        }
                    )
        read_satisfied = False
        read_failure_reason = None
        if checkpoint_read_pending_before is not None:
            read_failure_reason = filesystem_checkpoint_read_failure_reason(
                read_receipt,
                checkpoint_read_pending_before,
            )
            read_satisfied = read_failure_reason is None
            wrapper_evidence.update(
                {
                    "checkpoint_read_required": True,
                    "checkpoint_read_satisfied": read_satisfied,
                    "checkpoint_read_retry_pending": bool(
                        not read_satisfied and not bool(response["done"])
                    ),
                    "checkpoint_read_failure_reason": read_failure_reason,
                    "checkpoint_read_expected_size_bytes": (
                        checkpoint_read_pending_before.get("size_bytes")
                    ),
                    "checkpoint_read_expected_sha256": (
                        checkpoint_read_pending_before.get("sha256")
                    ),
                }
            )
            if read_satisfied or bool(response["done"]):
                self._pending_checkpoint_read = None
                self._pending_checkpoint_read_framing = None
        policy_state = (
            build_filesystem_checkpoint_read_retry_observation(
                read_failure_reason or "checkpoint_read_not_observed"
            )
            if checkpoint_read_pending_before is not None
            and not read_satisfied
            and not bool(response["done"])
            else str(response["observation"])
        )
        context_transition = None
        if (
            checkpoint_read_pending_before is not None
            and not read_satisfied
            and not bool(response["done"])
        ):
            if checkpoint_read_framing_before is None:
                raise RuntimeError(
                    "LiteResearcher pending checkpoint read lost its trusted framing"
                )
            context_transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=build_post_checkpoint_read_retry_context(
                    checkpoint_read_framing_before,
                    checkpoint_read_pending_before,
                    read_failure_reason or "checkpoint_read_not_observed",
                ),
            )
        return StepOutput(
            state=policy_state,
            reward=float(response["reward"]),
            done=bool(response["done"]),
            info=build_task_neutral_transition_info(
                env_info=response_info,
                action_submission=action_submission,
                native_step_before=native_before,
                native_step_after=self._native_call_count,
                native_call_count_before=native_before,
                native_call_count_after=self._native_call_count,
                context_epoch_before=context_before,
                context_epoch_after=self._context_epoch,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=policy_before,
                policy_step_after=self._policy_step_count,
                context_transition=context_transition,
                wrapper_evidence=wrapper_evidence,
            ),
        )

    def _complete_context_compaction(self, action: str) -> StepOutput:
        native_output = self._step_native_policy_action(action)
        self._selected_policy_control = None
        info = dict(native_output.info)
        env_info = info.get("env_info", {})
        server_wrapper = (
            env_info.get("wrapper_evidence", {})
            if isinstance(env_info, Mapping)
            else {}
        )
        receipt_value = (
            server_wrapper.get("filesystem_checkpoint")
            if isinstance(server_wrapper, Mapping)
            else None
        )
        checkpoint_receipt = normalize_filesystem_checkpoint_receipt(receipt_value)
        persisted = filesystem_checkpoint_write_succeeded(checkpoint_receipt)
        checkpoint_failure_reason = filesystem_checkpoint_failure_reason(
            checkpoint_receipt
        )
        self._checkpoint_retry_pending = bool(not persisted and not native_output.done)
        policy_observation = (
            build_filesystem_checkpoint_retry_observation(
                checkpoint_failure_reason or "unknown_checkpoint_failure"
            )
            if self._checkpoint_retry_pending
            else native_output.state
        )

        context_transition = None
        checkpoint_framing_sha256 = None
        if persisted and not native_output.done:
            framing = self._immutable_policy_context
            if framing is None:
                raise RuntimeError("LiteResearcher compaction lost its task framing")
            checkpoint_framing_sha256 = filesystem_checkpoint_framing_sha256(
                framing
            )
            replacement = build_post_checkpoint_context(
                framing, checkpoint_receipt
            )
            self._context_epoch += 1
            self._pending_checkpoint_read = dict(checkpoint_receipt)
            self._pending_checkpoint_read_framing = deepcopy(framing)
            context_transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=replacement,
            )
        elif native_output.done:
            self._pending_checkpoint_read = None
            self._pending_checkpoint_read_framing = None

        return StepOutput(
            state=policy_observation,
            reward=native_output.reward,
            done=native_output.done,
            info=build_task_neutral_transition_info(
                env_info=env_info if isinstance(env_info, Mapping) else {},
                action_submission=info.get(
                    "action_submission", {"raw_policy_output": action}
                ),
                native_step_before=info.get("native_step_before"),
                native_step_after=info.get("native_step_after"),
                native_call_count_before=info.get("native_call_count_before"),
                native_call_count_after=info.get("native_call_count_after"),
                context_epoch_before=info.get("context_epoch_before"),
                context_epoch_after=self._context_epoch,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=info.get("policy_step_before"),
                policy_step_after=info.get("policy_step_after"),
                context_transition=context_transition,
                wrapper_evidence={
                    "event": "context_compaction",
                    "workspace_continuity_id": self.env_id,
                    "native_environment_call_count": 1,
                    "continuation_path": FILESYSTEM_CHECKPOINT_PATH,
                    "continuation_max_bytes": FILESYSTEM_CHECKPOINT_MAX_BYTES,
                    "continuation_persisted": persisted,
                    "checkpoint_receipt": checkpoint_receipt,
                    "checkpoint_failure_reason": checkpoint_failure_reason,
                    "context_replaced": bool(persisted and not native_output.done),
                    "retry_pending": self._checkpoint_retry_pending,
                    "checkpoint_retry_observation_bounded": (
                        self._checkpoint_retry_pending
                    ),
                    "preserved_policy_output": persisted,
                    "preserved_native_observation": persisted,
                    "checkpoint_action_in_successor_context": False,
                    "checkpoint_observation_in_successor_context": False,
                    "checkpoint_content_in_successor_context": False,
                    "checkpoint_framing_sha256": checkpoint_framing_sha256,
                    "checkpoint_read_required_after": bool(
                        persisted and not native_output.done
                    ),
                    "server_wrapper_evidence": (
                        dict(server_wrapper)
                        if isinstance(server_wrapper, Mapping)
                        else {}
                    ),
                },
            ),
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._request(
            "POST", "reset", json={"id": self.env_id, "data_idx": idx}
        )
        self.info = response
        self._reset_policy_transition_state()
        return response

    def finalize_policy_horizon(self) -> StepOutput:
        return StepOutput(
            state="LiteResearcher policy-turn budget exhausted without an accepted answer.",
            reward=0.0,
            done=True,
            info=build_task_neutral_transition_info(
                env_info={
                    **dict(self.info.get("info", {})),
                    "status": "max_policy_steps_exhausted",
                    "episode_success": False,
                    "sample_excluded": False,
                },
                action_submission={"control_action": "horizon"},
                native_step_before=self._native_call_count,
                native_step_after=self._native_call_count,
                native_call_count_before=self._native_call_count,
                native_call_count_after=self._native_call_count,
                context_epoch_before=self._context_epoch,
                context_epoch_after=self._context_epoch,
                session_epoch_before=0,
                session_epoch_after=0,
                policy_step_before=self._policy_step_count,
                policy_step_after=self._policy_step_count,
                wrapper_evidence={
                    "event": "horizon_finalization",
                    "outcome": "max_rounds",
                },
            ),
        )

    def close(self) -> bool:
        value = self._request_json("POST", "close", json={"id": self.env_id})
        if value is not True:
            raise requests.RequestException(
                "LiteResearcher POST /close did not return true"
            )
        return True

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        value = self._request_json(method, path, **kwargs)
        if not isinstance(value, dict):
            raise requests.RequestException(
                f"LiteResearcher {method} /{path} returned a non-object response"
            )
        return value

    def _request_json(self, method: str, path: str, **kwargs) -> Any:
        response = requests.request(
            method,
            f"{self.env_server_base}/{path}",
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code != 200:
            raise requests.RequestException(
                f"LiteResearcher {method} /{path} failed: "
                f"status={response.status_code} body={response.text[-1000:]}"
            )
        return response.json()


def _copy_policy_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"policy message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"policy message {index} has invalid role: {role!r}")
        if not isinstance(content, str):
            raise TypeError(f"policy message {index} content must be text")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("policy context must not be empty")
    return normalized


class LiteResearcherTask(BaseTask):
    env_client_cls = LiteResearcherEnvClient
    env_name = "LiteResearcher"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
