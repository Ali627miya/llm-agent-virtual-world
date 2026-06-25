from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .entities import Direction, Tile, WorldObject


@dataclass
class WorldConfig:
    width: int = 8
    height: int = 8
    max_steps: int = 40


@dataclass
class StepResult:
    success: bool
    message: str
    done: bool = False
    reward: float = 0.0


@dataclass
class GridWorld:
    """A small 2D grid world with walls, a key, and a locked exit door."""

    config: WorldConfig = field(default_factory=WorldConfig)
    agent_x: int = 1
    agent_y: int = 1
    facing: Direction = Direction.EAST
    inventory: list[str] = field(default_factory=list)
    steps: int = 0
    door_locked: bool = True
    goal_completed: bool = False
    last_message: str = "You wake up in a dim corridor. Find the key and unlock the exit."

    def __post_init__(self) -> None:
        self.tiles: list[list[Tile]] = [
            [Tile.FLOOR for _ in range(self.config.width)]
            for _ in range(self.config.height)
        ]
        self.objects: dict[tuple[int, int], WorldObject] = {}
        self._build_default_map()

    def _build_default_map(self) -> None:
        for x in range(self.config.width):
            self.tiles[0][x] = Tile.WALL
            self.tiles[self.config.height - 1][x] = Tile.WALL
        for y in range(self.config.height):
            self.tiles[y][0] = Tile.WALL
            self.tiles[y][self.config.width - 1] = Tile.WALL

        for x in range(2, 6):
            self.tiles[3][x] = Tile.WALL
        self.tiles[3][6] = Tile.FLOOR

        self.tiles[5][2] = Tile.WALL
        self.tiles[5][3] = Tile.WALL

        self.tiles[self.config.height - 2][self.config.width - 2] = Tile.EXIT
        self.objects[(5, 1)] = WorldObject.key()
        self.objects[(2, 5)] = WorldObject.red_cube()

    def reset(self) -> None:
        self.__init__(config=self.config)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def tile_at(self, x: int, y: int) -> Tile | None:
        if not self.in_bounds(x, y):
            return None
        return self.tiles[y][x]

    def object_at(self, x: int, y: int) -> WorldObject | None:
        return self.objects.get((x, y))

    def front_cell(self) -> tuple[int, int]:
        dx, dy = self.facing.delta()
        return self.agent_x + dx, self.agent_y + dy

    def can_enter(self, x: int, y: int) -> bool:
        tile = self.tile_at(x, y)
        if tile is None or tile == Tile.WALL:
            return False
        obj = self.object_at(x, y)
        if obj and obj.blocks_movement and not obj.collectible:
            return False
        if obj and obj.blocks_movement and obj.collectible:
            return False
        return True

    def render_ascii(self) -> str:
        facing_glyph = {"north": "^", "east": ">", "south": "v", "west": "<"}
        rows: list[str] = []
        for y in range(self.config.height):
            row = []
            for x in range(self.config.width):
                if x == self.agent_x and y == self.agent_y:
                    row.append(facing_glyph[self.facing.label])
                    continue
                tile = self.tiles[y][x]
                if tile == Tile.WALL:
                    row.append("#")
                elif tile == Tile.EXIT:
                    row.append("E" if self.door_locked else "X")
                else:
                    obj = self.object_at(x, y)
                    if obj and obj.kind == "key":
                        row.append("K")
                    elif obj and obj.kind == "cube":
                        row.append("R")
                    else:
                        row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def apply_action(self, action: str) -> StepResult:
        if self.goal_completed:
            return StepResult(True, "Goal already completed.", done=True)

        self.steps += 1
        action = action.strip().lower()

        handlers = {
            "move_forward": self._move_forward,
            "turn_left": self._turn_left,
            "turn_right": self._turn_right,
            "pick_up": self._pick_up,
            "use": self._use,
            "look": self._look,
        }

        if action not in handlers:
            return StepResult(
                False,
                f"Unknown action '{action}'. Valid actions: {', '.join(handlers)}.",
            )

        result = handlers[action]()
        self.last_message = result.message

        if self.steps >= self.config.max_steps and not self.goal_completed:
            result.done = True
            result.message += " Step limit reached."

        return result

    def _move_forward(self) -> StepResult:
        x, y = self.front_cell()
        if not self.can_enter(x, y):
            return StepResult(False, "Blocked. Something is in the way.")
        self.agent_x, self.agent_y = x, y
        return StepResult(True, f"Moved forward to ({x}, {y}).")

    def _turn_left(self) -> StepResult:
        self.facing = self.facing.turn_left()
        return StepResult(True, f"Turned left; now facing {self.facing.label}.")

    def _turn_right(self) -> StepResult:
        self.facing = self.facing.turn_right()
        return StepResult(True, f"Turned right; now facing {self.facing.label}.")

    def _pick_up(self) -> StepResult:
        x, y = self.front_cell()
        obj = self.object_at(x, y)
        if obj is None:
            return StepResult(False, "Nothing to pick up in front of you.")
        if not obj.collectible:
            return StepResult(False, f"You cannot pick up the {obj.name}.")
        self.inventory.append(obj.name)
        del self.objects[(x, y)]
        return StepResult(True, f"Picked up the {obj.name}.")

    def _use(self) -> StepResult:
        x, y = self.front_cell()
        tile = self.tile_at(x, y)
        obj = self.object_at(x, y)

        if tile == Tile.EXIT and self.door_locked:
            if "golden key" in self.inventory:
                self.door_locked = False
                self.goal_completed = True
                return StepResult(
                    True,
                    "You unlocked the exit door. Mission complete!",
                    done=True,
                    reward=1.0,
                )
            return StepResult(False, "The exit door is locked. You need a key.")

        if obj and obj.usable:
            return StepResult(False, f"The {obj.name} cannot be used here.")

        return StepResult(False, "Nothing usable in front of you.")

    def _look(self) -> StepResult:
        return StepResult(True, "You study your surroundings carefully.")

    def snapshot(self) -> dict[str, Any]:
        return {
            "position": {"x": self.agent_x, "y": self.agent_y},
            "facing": self.facing.label,
            "inventory": list(self.inventory),
            "door_locked": self.door_locked,
            "goal_completed": self.goal_completed,
            "steps": self.steps,
            "max_steps": self.config.max_steps,
        }
