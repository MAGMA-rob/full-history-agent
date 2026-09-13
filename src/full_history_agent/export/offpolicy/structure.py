from dataclasses import dataclass
from typing import Any, Literal

from magma_core.domain import Call

from magma_offpolicy_gen.agents.execution_context import TraceExecutionContext


@dataclass(frozen=True)
class HRInstruction:
    author: Literal["USER", "SYSTEM"]
    content: str


@dataclass(frozen=True)
class HRHistoryEntry:
    author: Literal["USER", "SYSTEM", "MODEL"]
    content: str


@dataclass
class HRProjectionContext:
    execution: TraceExecutionContext
    history: list[HRHistoryEntry]
    pending_instruction: HRInstruction | None = None

    def set_instruction(self, instruction: HRInstruction) -> None:
        if self.pending_instruction is not None:
            self.history.append(HRHistoryEntry(
                author=self.pending_instruction.author,
                content=self.pending_instruction.content,
            ))
        self.pending_instruction = instruction


@dataclass(frozen=True)
class HRActionProjection:
    attributes: dict[str, Any]
    instruction: HRInstruction
    history: tuple[HRHistoryEntry, ...]
    calls: tuple[Call, ...] | None = None
    message: str | None = None
