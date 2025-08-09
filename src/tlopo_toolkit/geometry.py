import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import Literal
from typing import NamedTuple
from typing import Type
from typing import TypeVar

from typing_extensions import Self

from pydantic import AfterValidator
from pydantic import BaseModel

T: TypeVar = TypeVar("T")


@dataclass(frozen=True)
class Point:
    __slots__: ClassVar[list[str]] = ["x", "y"]

    x: float
    y: float


class Rect(NamedTuple):
    """Represents a rectangle on the board."""
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class _Hex(Generic[T]):
    """Represents a hexagon coordinate."""
    __slots__: ClassVar[tuple[str, str, str]] = ("q", "r", "s")

    q: T
    r: T
    s: T

    def __post_init__(self) -> None:
        if round(self.q + self.r + self.s) != 0:
            raise ValueError("q + r + s must be 0")

    def __add__(self, other: Self) -> Self:
        return _Hex(self.q + other.q, self.r + other.r, self.s + other.s)

    def __sub__(self, other: Self) -> Self:
        return _Hex(self.q - other.q, self.r - other.r, self.s - other.s)


@dataclass(frozen=True)
class Hex(_Hex[int]):
    """Represents a Hex."""

    def __add__(self, other: Self) -> Self:
        return Hex(self.q + other.q, self.r + other.r, self.s + other.s)

    def __sub__(self, other: Self) -> Self:
        return Hex(self.q - other.q, self.r - other.r, self.s - other.s)



@dataclass(frozen=True)
class HexFractional(_Hex[float]):
    """Represents a Fractional Hex."""

    def __add__(self, other: Self) -> Self:
        return HexFractional(self.q + other.q, self.r + other.r, self.s + other.s)

    def __sub__(self, other: Self) -> Self:
        return HexFractional(self.q - other.q, self.r - other.r, self.s - other.s)


def hex_add(a: _Hex, b: Hex):
    return Hex(a.q + b.q, a.r + b.r, a.s + b.s)


def hex_subtract(a, b):
    return Hex(a.q - b.q, a.r - b.r, a.s - b.s)


def hex_scale(a, k):
    return Hex(a.q * k, a.r * k, a.s * k)


def hex_rotate_left(a):
    return Hex(-a.s, -a.q, -a.r)


def hex_rotate_right(a):
    return Hex(-a.r, -a.s, -a.q)


hex_directions: list[Hex] = [Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1), Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)]


def hex_direction(direction: int) -> Hex:
    return hex_directions[direction]


def hex_neighbor(hex: Hex, direction: int) -> Hex:
    return hex + hex_direction(direction)


hex_diagonals: list[Hex] = [Hex(2, -1, -1), Hex(1, -2, 1), Hex(-1, -1, 2), Hex(-2, 1, 1), Hex(-1, 2, -1), Hex(1, 1, -2)]


def hex_diagonal_neighbor(hex: _Hex, direction: int) -> _Hex:
    return hex + hex_diagonals[direction]


def hex_length(hex: Hex) -> int:
    return (abs(hex.q) + abs(hex.r) + abs(hex.s)) // 2


def hex_distance(a: Hex, b: Hex) -> int:
    return hex_length(a - b)


def hex_round(h: _Hex) -> Hex:
    qi: int = int(round(h.q))
    ri: int = int(round(h.r))
    si: int = int(round(h.s))
    q_diff: int = abs(qi - h.q)
    r_diff: int = abs(ri - h.r)
    s_diff: int = abs(si - h.s)
    if q_diff > r_diff and q_diff > s_diff:
        qi = -ri - si
    else:
        if r_diff > s_diff:
            ri = -qi - si
        else:
            si = -qi - ri
    return Hex(qi, ri, si)


def hex_lerp(a, b, t):
    return Hex(a.q * (1.0 - t) + b.q * t, a.r * (1.0 - t) + b.r * t, a.s * (1.0 - t) + b.s * t)


def hex_linedraw(a, b):
    N = hex_distance(a, b)
    a_nudge = Hex(a.q + 1e-06, a.r + 1e-06, a.s - 2e-06)
    b_nudge = Hex(b.q + 1e-06, b.r + 1e-06, b.s - 2e-06)
    results = []
    step = 1.0 / max(N, 1)
    for i in range(0, N + 1):
        results.append(hex_round(hex_lerp(a_nudge, b_nudge, step * i)))
    return results


EVEN: Literal[1] = 1
ODD: Literal[-1] = -1


@dataclass(frozen=True)
class OffsetCoord:
    __slots__: ClassVar[list[str]] = ["col", "row"]

    col: int
    row: int


@dataclass(frozen=True)
class DoubledCoord:
    __slots__: ClassVar[list[str]] = ["col", "row"]

    col: int
    row: int


def qoffset_from_cube(offset: Literal[-1, 1], h: Hex) -> OffsetCoord:
    parity = h.q & 1
    col: int = h.q
    row: int = h.r + (h.q + offset * parity) // 2
    return OffsetCoord(col, row)


def qoffset_to_cube(offset: Literal[-1, 1], h: OffsetCoord) -> Hex:
    parity = h.col & 1
    q: int = h.col
    r: int = h.row - (h.col + offset * parity) // 2
    s: int = -q - r
    return Hex(q, r, s)


def roffset_from_cube(offset: Literal[-1, 1], h: Hex) -> OffsetCoord:
    parity = h.r & 1
    col: int = h.q + (h.r + offset * parity) // 2
    row: int = h.r
    return OffsetCoord(col, row)


def roffset_to_cube(offset: Literal[-1, 1], h: OffsetCoord) -> Hex:
    parity = h.row & 1
    q: int = h.col - (h.row + offset * parity) // 2
    r: int = h.row
    s: int = -q - r
    return Hex(q, r, s)


def qoffset_from_qdoubled(offset: Literal[-1, 1], h: DoubledCoord) -> OffsetCoord:
    parity = h.col & 1
    return OffsetCoord(h.col, (h.row + offset * parity) // 2)


def qoffset_to_qdoubled(offset: Literal[-1, 1], h: OffsetCoord) -> DoubledCoord:
    parity = h.col & 1
    return DoubledCoord(h.col, 2 * h.row - offset * parity)


def roffset_from_rdoubled(offset: Literal[-1, 1], h: DoubledCoord) -> OffsetCoord:
    parity = h.row & 1
    return OffsetCoord((h.col + offset * parity) // 2, h.row)


def roffset_to_rdoubled(offset: Literal[-1, 1], h: OffsetCoord) -> DoubledCoord:
    parity = h.row & 1
    return DoubledCoord(2 * h.col - offset * parity, h.row)



def qdoubled_from_cube(h: Hex) -> DoubledCoord:
    col: int = h.q
    row: int = 2 * h.r + h.q
    return DoubledCoord(col, row)


def qdoubled_to_cube(h: DoubledCoord) -> Hex:
    q: int = h.col
    r: int = (h.row - h.col) // 2
    s: int = -q - r
    return Hex(q, r, s)


def rdoubled_from_cube(h: Hex) -> DoubledCoord:
    col: int = 2 * h.q + h.r
    row: int = h.r
    return DoubledCoord(col, row)


def rdoubled_to_cube(h: DoubledCoord) -> Hex:
    q: int = (h.col - h.row) // 2
    r: int = h.row
    s: int = -q - r
    return Hex(q, r, s)


@dataclass(frozen=True)
class Orientation:
    __slots__: ClassVar[list[str]] = ["f0", "f1", "f2", "f3", "b0", "b1", "b2", "b3", "start_angle"]

    f0: float
    f1: float
    f2: float
    f3: float
    b0: float
    b1: float
    b2: float
    b3: float
    start_angle: float


@dataclass(frozen=True)
class Layout:
    __slots__: ClassVar[list[str]] = ["orientation", "size", "origin"]

    orientation: Orientation
    size: Point
    origin: Point


ORIENTATION_POINTY: Orientation = Orientation(
    math.sqrt(3.0),
    math.sqrt(3.0) / 2.0,
    0.0,
    3.0 / 2.0,
    math.sqrt(3.0) / 3.0,
    -1.0 / 3.0,
    0.0,
    2.0 / 3.0,
    0.5
)
ORIENTATION_FLAT: Orientation = Orientation(
    3.0 / 2.0,
    0.0,
    math.sqrt(3.0) / 2.0,
    math.sqrt(3.0),
    2.0 / 3.0,
    0.0,
    -1.0 / 3.0,
    math.sqrt(3.0) / 3.0,
    0.0
)


def hex_to_pixel(layout: Layout, h: Hex) -> Point:
    orientation: Orientation = layout.orientation
    size: Point = layout.size
    origin: Point = layout.origin
    x: float = (orientation.f0 * h.q + orientation.f1 * h.r) * size.x
    y: float = (orientation.f2 * h.q + orientation.f3 * h.r) * size.y
    return Point(x + origin.x, y + origin.y)


def pixel_to_hex_fractional(layout: Layout, p: Point) -> HexFractional:
    orientation = layout.orientation
    size = layout.size
    origin = layout.origin
    pt = Point((p.x - origin.x) / size.x, (p.y - origin.y) / size.y)
    q = orientation.b0 * pt.x + orientation.b1 * pt.y
    r = orientation.b2 * pt.x + orientation.b3 * pt.y
    return HexFractional(q, r, -q - r)


def pixel_to_hex_rounded(layout: Layout, p: Point) -> Hex:
    return hex_round(pixel_to_hex_fractional(layout, p))


def hex_corner_offset(layout: Layout, corner: int) -> Point:
    orientation: Orientation = layout.orientation
    size: Point = layout.size
    angle: float = 2.0 * math.pi * (orientation.start_angle - corner) / 6.0
    return Point(size.x * math.cos(angle), size.y * math.sin(angle))


def polygon_corners(layout: Layout, h: Hex) -> list[Point]:
    corners: list[Point] = []
    center: Point = hex_to_pixel(layout, h)
    for i in range(0, 6):
        offset = hex_corner_offset(layout, i)
        corners.append(Point(center.x + offset.x, center.y + offset.y))
    return corners
