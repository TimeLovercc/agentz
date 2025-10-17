from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.context.conversation import ConversationState, create_conversation_state


class WriterInput(BaseModel):
    user_prompt: str = Field()
    data_path: str = Field()
    findings: str = Field()
    guidelines_block: str = Field(default="")
    gap: str = Field(default="")


def make_state() -> ConversationState:
    # create_conversation_state wires the required private iteration model
    return create_conversation_state({})


def test_runtime_context_aliases_and_overrides() -> None:
    state = make_state()
    payload = WriterInput(
        user_prompt="Prompt",
        data_path="/tmp/data.csv",
        findings="Found something",
        guidelines_block="Use markdown",
        gap="Missing evaluation",
    )
    placeholders = {"input", "task", "gap", "instructions", "user_prompt", "data_path", "findings", "guidelines_block"}

    state.update_runtime_context(current_input="Current input", payload=payload, placeholders=placeholders)

    # Aliases use current input by default
    assert state.resolve_placeholder("input") == "Current input"
    assert state.resolve_placeholder("task") == "Current input"
    assert state.resolve_placeholder("instructions") == "Current input"

    # Structured payload fields override aliases where present
    assert state.resolve_placeholder("gap") == "Missing evaluation"
    assert state.resolve_placeholder("user_prompt") == "Prompt"
    assert state.resolve_placeholder("data_path") == "/tmp/data.csv"
    assert state.resolve_placeholder("findings") == "Found something"
    assert state.resolve_placeholder("guidelines_block") == "Use markdown"


def test_runtime_context_persists_pipeline_defaults() -> None:
    state = make_state()
    state.set_runtime_value("max_minutes", 7)
    assert state.resolve_placeholder("max_minutes") == "7"
