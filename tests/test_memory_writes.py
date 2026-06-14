"""Memory write paths: token-budget rollover + conversation-end episodic embed.

Covers the two design decisions in ROADMAP/CLAUDE:
  - Working-memory rollover is *token-budget-driven* (not a fixed turn count).
  - Episodic memory embeds *one point per conversation* at end, never per turn.
"""

from __future__ import annotations

from agent_kit.config import AgentKitConfig, WorkingMemoryConfig
from agent_kit.memory.working import RolledSummary, WorkingMemory
from agent_kit.stores.memory_session import InMemorySessionStore
from agent_kit.stores.types import SessionState, Turn

from tests.conftest import FakeLLM, ScriptedTurn, make_service


# ------------------------------------------------------------ rollover (working)


async def _seed(store: InMemorySessionStore, convo: str, user: str, texts: list[str]):
    await store.save(convo, SessionState(user_id=user))
    for text in texts:
        await store.append_turn(convo, Turn(role="user", text=text))


async def test_rollover_summarizes_and_evicts_oldest_when_over_budget():
    store = InMemorySessionStore()
    # 4-token budget (≈16 chars); each seeded turn is ~16 chars → 4 tokens.
    cfg = WorkingMemoryConfig(buffer_token_budget=4)
    llm = FakeLLM(invoke_parsed=RolledSummary(summary="rolled summary"))
    wm = WorkingMemory(store, cfg, llm=llm)
    await _seed(store, "c1", "alice", [f"message number {i}" for i in range(5)])

    await wm.maybe_rollover("c1", "alice")

    state = await store.load("c1", "alice")
    assert state.rolling_summary == "rolled summary"
    # Only the newest turn fits the budget; the older four were folded in + dropped.
    assert [t.text for t in state.working_buffer] == ["message number 4"]


async def test_no_rollover_when_under_budget():
    store = InMemorySessionStore()
    cfg = WorkingMemoryConfig(buffer_token_budget=10_000)
    llm = FakeLLM(invoke_parsed=RolledSummary(summary="should not be written"))
    wm = WorkingMemory(store, cfg, llm=llm)
    await _seed(store, "c1", "alice", ["hi", "there"])

    await wm.maybe_rollover("c1", "alice")

    state = await store.load("c1", "alice")
    assert state.rolling_summary == ""  # summarizer never invoked
    assert len(state.working_buffer) == 2


async def test_rollover_is_noop_without_an_llm_and_loses_no_turns():
    store = InMemorySessionStore()
    cfg = WorkingMemoryConfig(buffer_token_budget=1)  # would trip if it could
    wm = WorkingMemory(store, cfg)  # no llm wired
    await _seed(store, "c1", "alice", ["a long enough message", "another long message"])

    await wm.maybe_rollover("c1", "alice")

    state = await store.load("c1", "alice")
    assert len(state.working_buffer) == 2  # nothing summarized, nothing dropped
    assert state.rolling_summary == ""


def test_needs_rollover_is_token_based():
    cfg = WorkingMemoryConfig(buffer_token_budget=4)
    wm = WorkingMemory(InMemorySessionStore(), cfg)
    small = [Turn(role="user", text="hi")]  # ~0 tokens
    big = [Turn(role="user", text="x" * 100)]  # 25 tokens > 4
    assert wm.needs_rollover(small) is False
    assert wm.needs_rollover(big) is True


# ----------------------------------------------------- episodic (conversation end)


async def _run(agent, user="alice", convo="c1", msg="hi"):
    async for _ in agent.run_turn(user, convo, msg):
        pass
    await agent.drain()


async def test_no_episodic_point_written_per_turn():
    base = AgentKitConfig()
    service, _ = make_service(
        base, turns=[ScriptedTurn(text_chunks=["one"]), ScriptedTurn(text_chunks=["two"])]
    )
    await _run(service.agent, msg="first")
    await _run(service.agent, msg="second")
    # Turns accumulate in working memory but nothing is embedded mid-conversation.
    assert service.stores.vectors._points == {}


async def test_end_conversation_writes_exactly_one_point():
    base = AgentKitConfig()
    service, _ = make_service(
        base, turns=[ScriptedTurn(text_chunks=["hello there"])]
    )
    await _run(service.agent, msg="I'm Sam, I like aisle seats")

    await service.agent.end_conversation("alice", "c1")

    points = list(service.stores.vectors._points.values())
    assert len(points) == 1
    payload = points[0].payload
    assert payload["user_id"] == "alice"
    assert payload["conversation_id"] == "c1"
    assert payload["kind"] == "conversation"
    # The single point covers the whole conversation (both turns folded in).
    assert "I'm Sam, I like aisle seats" in payload["text"]
    assert "hello there" in payload["text"]


async def test_end_conversation_for_non_owner_is_noop():
    base = AgentKitConfig()
    service, _ = make_service(base, turns=[ScriptedTurn(text_chunks=["hi"])])
    await _run(service.agent, user="alice", convo="c1", msg="hello")

    # bob does not own c1 → best-effort cleanup writes nothing and does not raise.
    await service.agent.end_conversation("bob", "c1")

    assert service.stores.vectors._points == {}
