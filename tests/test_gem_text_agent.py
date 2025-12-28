"""Tests for GEMTextAgent parsing and state updates."""

from agents.gem_text_agent import GEMTextAgent, extract_last_boxed
from rllm.agents.agent import Action  # type: ignore


def test_extract_last_boxed():
    assert extract_last_boxed("foo \\boxed{3} bar \\boxed{7}") == "7"
    assert extract_last_boxed("no boxes here") == "no boxes here"


def test_update_flow():
    agent = GEMTextAgent(system_prompt="sys", max_steps=5)
    agent.update_from_env("hello", reward=0.0, done=False, info={})
    action = agent.update_from_model("try \\boxed{5}")

    assert isinstance(action, Action)
    assert action.action == "5"
    state = agent.get_current_state()
    assert state.action.action == "5"
    assert state.observation == "hello"
    assert len(agent.chat_completions) >= 2

