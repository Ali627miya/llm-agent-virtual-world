from src.world.grid_world import GridWorld
from src.llm.client import ScriptedPolicy, key_and_door_script
from src.harness.agent_loop import AgentHarness


def test_scripted_agent_completes_goal():
    world = GridWorld()
    harness = AgentHarness(world=world, policy=ScriptedPolicy(key_and_door_script()))
    log = harness.run()
    assert world.goal_completed, "Scripted policy should unlock the exit"
    assert log.completed


def test_invalid_action_returns_feedback():
    world = GridWorld()
    result = world.apply_action("fly")
    assert not result.success
    assert "Unknown action" in result.message


def test_pick_up_requires_item_in_front():
    world = GridWorld()
    result = world.apply_action("pick_up")
    assert not result.success
