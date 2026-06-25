from __future__ import annotations

import json
from typing import Any

from src.world.grid_world import GridWorld
from src.world.entities import Direction, Tile


DEFAULT_GOAL = (
    "Find the golden key somewhere in the maze, reach the exit door (marked E), "
    "and use the key to unlock it before you run out of steps."
)


def _describe_cell(world: GridWorld, x: int, y: int) -> dict[str, Any]:
    tile = world.tile_at(x, y)
    if tile is None:
        return {"in_bounds": False, "tile": "void", "object": None, "passable": False}

    obj = world.object_at(x, y)
    passable = world.can_enter(x, y) if tile != Tile.WALL else False

    description: dict[str, Any] = {
        "in_bounds": True,
        "tile": tile.value,
        "passable": passable,
        "object": None,
    }

    if obj:
        description["object"] = {
            "kind": obj.kind,
            "name": obj.name,
            "collectible": obj.collectible,
        }

    if tile == Tile.EXIT:
        description["door_locked"] = world.door_locked

    return description


def _raycast_front(world: GridWorld, max_distance: int = 3) -> list[dict[str, Any]]:
    dx, dy = world.facing.delta()
    x, y = world.agent_x, world.agent_y
    seen: list[dict[str, Any]] = []

    for distance in range(1, max_distance + 1):
        tx, ty = x + dx * distance, y + dy * distance
        cell = _describe_cell(world, tx, ty)
        cell["distance"] = distance
        seen.append(cell)
        if not cell["in_bounds"] or not cell["passable"]:
            break

    return seen


def _relative_cell(world: GridWorld, side: str) -> dict[str, Any]:
    facing = world.facing
    if side == "left":
        direction = facing.turn_left()
    elif side == "right":
        direction = facing.turn_right()
    else:
        direction = facing.turn_left().turn_left()

    dx, dy = direction.delta()
    return _describe_cell(world, world.agent_x + dx, world.agent_y + dy)


def build_observation(
    world: GridWorld,
    goal: str = DEFAULT_GOAL,
    last_feedback: str | None = None,
) -> dict[str, Any]:
    """Build a structured, ego-centric observation for the agent."""
    return {
        "step": world.steps,
        "max_steps": world.config.max_steps,
        "goal": goal,
        "position": {"x": world.agent_x, "y": world.agent_y},
        "facing": world.facing.label,
        "inventory": list(world.inventory),
        "task_status": {
            "has_key": "golden key" in world.inventory,
            "exit_unlocked": not world.door_locked,
            "goal_completed": world.goal_completed,
        },
        "sensors": {
            "front_scan": _raycast_front(world),
            "left": _relative_cell(world, "left"),
            "right": _relative_cell(world, "right"),
            "behind": _relative_cell(world, "behind"),
        },
        "map_hint": (
            "Legend: # wall, . floor, K key, R red cube (blocks path), "
            "E locked exit, X unlocked exit, agent shown as direction arrow."
        ),
        "last_feedback": last_feedback or world.last_message,
        "valid_actions": [
            "move_forward",
            "turn_left",
            "turn_right",
            "pick_up",
            "use",
            "look",
        ],
    }


def format_observation_for_llm(observation: dict[str, Any]) -> str:
    return json.dumps(observation, indent=2)
