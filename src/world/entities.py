from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def turn_left(self) -> Direction:
        return Direction((self.value - 1) % 4)

    def turn_right(self) -> Direction:
        return Direction((self.value + 1) % 4)

    def delta(self) -> tuple[int, int]:
        deltas = {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0),
        }
        return deltas[self]

    @property
    def label(self) -> str:
        return self.name.lower()


class Tile(Enum):
    FLOOR = "floor"
    WALL = "wall"
    DOOR = "door"
    EXIT = "exit"


@dataclass(frozen=True)
class WorldObject:
    kind: str
    name: str
    collectible: bool = False
    usable: bool = False
    blocks_movement: bool = False

    @staticmethod
    def key() -> WorldObject:
        return WorldObject(
            kind="key",
            name="golden key",
            collectible=True,
            usable=True,
        )

    @staticmethod
    def red_cube() -> WorldObject:
        return WorldObject(
            kind="cube",
            name="red cube",
            collectible=True,
            blocks_movement=True,
        )
