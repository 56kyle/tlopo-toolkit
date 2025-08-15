import math
from dataclasses import dataclass
from typing import ClassVar
from typing import Generic
from typing import Literal
from typing import Optional
from typing import TypeVar

import cv2
import numpy as np
from shapely import Point
from typing_extensions import Self


T: TypeVar = TypeVar("T")


@dataclass(frozen=True)
class Rect:
    """Represents a Rectangle."""

    x: int
    y: int
    w: int
    h: int

    __slots__: ClassVar[list[str]] = ["x", "y", "w", "h"]

    @property
    def left(self):
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class Region:
    """Represents a Rectangle Region in a larger img."""

    img: np.ndarray
    rect: Rect

    __slots__: ClassVar[list[str]] = ["img", "rect"]

    def crop_absolute(self, rect: Rect) -> Self:
        """Crops the internal region relative to the img."""
        return Region(img=self.img, rect=rect)

    def crop_relative(self, rect: Rect) -> Self:
        """Crops the internal region to itself."""

        return Region(
            img=self.img,
            rect=Rect(
                x=self.rect.x + rect.x,
                y=self.rect.y + rect.y,
                w=rect.w,
                h=rect.h,
            ),
        )

    def export(self) -> np.ndarray:
        """Exports the internal region as a numpy array."""
        return np.copy(self.img[self.rect.top : self.rect.bottom, self.rect.left : self.rect.right])

    def show(self) -> None:
        new_img: np.ndarray = np.copy(self.img)
        if len(new_img.shape) == 2:
            new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)

        rect_img: np.ndarray = cv2.rectangle(
            new_img, (self.rect.x, self.rect.y), (self.rect.right, self.rect.bottom), (0, 255, 0), 1
        )
        cv2.imshow("img", rect_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
    qi: int = round(h.q)
    ri: int = round(h.r)
    si: int = round(h.s)
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
    __slots__: ClassVar[list[str]] = ["b0", "b1", "b2", "b3", "f0", "f1", "f2", "f3", "start_angle"]

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
    __slots__: ClassVar[list[str]] = ["orientation", "origin", "size"]

    orientation: Orientation
    size: Point
    origin: Point


ORIENTATION_POINTY: Orientation = Orientation(
    math.sqrt(3.0), math.sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0, math.sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5
)
ORIENTATION_FLAT: Orientation = Orientation(
    3.0 / 2.0, 0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0), 2.0 / 3.0, 0.0, -1.0 / 3.0, math.sqrt(3.0) / 3.0, 0.0
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


def polygon_lines(layout: Layout, h: Hex) -> set[Point]:
    """Returns a list of Points making up the lines of the hexagon.

    This is found by linearly interpolating the corners.
    """

    corners: list[Point] = polygon_corners(layout, h)
    lines: list[tuple[Point, Point]] = []
    for i in range(0, 6):
        lines.append((corners[i], corners[(i + 1) % 6]))

    points: set[Point] = set()

    for segment in lines:
        for point in linear_interpolate_points(segment[0], segment[1]):
            points.add(Point(point))

    return points


def linear_interpolate_points(p1: Point, p2: Point, num_points: Optional[int] = None) -> np.ndarray:
    """Linear interpolation between two points using NumPy, returning integer coordinates."""
    p1 = np.array([p1.x, p1.y])
    p2 = np.array([p2.x, p2.y])

    # If num_points not specified, use max distance for smooth interpolation
    if num_points is None:
        num_points: int = int(np.max(np.abs(p2 - p1)))

    # Handle case where points are the same
    if num_points == 0:
        return np.array([p1], dtype=int)

    # Create parameter array from 0 to 1
    t: np.ndarray = np.linspace(0, 1, num_points + 1)

    # Linear interpolation: p = p1 + t * (p2 - p1)
    points: np.ndarray = p1 + t[:, np.newaxis] * (p2 - p1)

    # Round and convert to integers
    return np.round(points).astype(int)
