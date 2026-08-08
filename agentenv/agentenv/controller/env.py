from abc import ABCMeta, abstractmethod
from typing import Mapping, Sequence

from .types import (
    ActionFormat,
    ConversationMessage,
    PolicyContextPressure,
    StepOutput,
)


class BaseEnvClient(metaclass=ABCMeta):
    _conversation_start: dict[ActionFormat, tuple[ConversationMessage]]

    def __init__(self, action_format: ActionFormat = "react") -> None:
        self.action_format = ActionFormat(action_format)

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total size of the environment.
        """

    @abstractmethod
    def observe(self) -> str:
        """
        Parse env server response and give a text message to prompt the LLM.
        """

    @abstractmethod
    def step(self, action) -> StepOutput:
        """
        Parse model output from the action and call the env server.
        """

    @abstractmethod
    def reset(self, idx: int) -> None:
        """
        Reset the environment.
        """

    def policy_turn_candidate(self) -> str | None:
        """Return a wrapper-owned control request that the runner may measure."""

        return None

    def policy_framing(self) -> Sequence[Mapping[str, str]] | None:
        """Return wrapper-owned immutable policy framing when one is configured.

        Most environments keep using the dataset conversation start. Wrappers
        with a formal runtime prompt can expose it here so dataset bootstrap and
        online rollout bind the same framing.
        """

        return None

    def normalize_initial_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> Sequence[Mapping[str, str]]:
        """Normalize the first online context before it is bound to the wrapper."""

        return messages

    def bind_policy_context(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        initial: bool = False,
    ) -> None:
        """Expose the current policy context to a wrapper state machine.

        ``initial=True`` binds the immutable task framing once per reset. Later
        calls expose the exact current prompt without changing that framing.
        """

        del messages, initial

    def prepare_policy_turn(
        self, pressure: PolicyContextPressure | None
    ) -> str | None:
        """Choose whether the next policy row uses the measured control request."""

        del pressure
        return None

    def finalize_policy_horizon(self) -> StepOutput | None:
        """Optionally grade a workspace when the shared policy budget expires.

        The method is a wrapper-owned lifecycle hook.  It does not consume a
        sampled policy row; the shared runner only applies the returned reward
        and receipt to the final row already sampled at the horizon.
        """

        return None
