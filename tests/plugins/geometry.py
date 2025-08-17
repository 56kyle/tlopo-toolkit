import pytest
from _pytest.fixtures import FixtureRequest

from tlopo_toolkit.geometry import Hex
from tlopo_toolkit.geometry import HexFractional


@pytest.fixture(scope="function")
def point(request: FixtureRequest) -> Point:
    return getattr(request, "param", Point(
        x=point__x,
        y=point__y
    ))


@pytest.fixture(scope="function")
def hex_basic(
    request: FixtureRequest,
    hex_basic__q: int,
    hex_basic__r: int,
    hex_basic__s: int
) -> Hex:
    return getattr(request, "param", Hex(
        q=hex_basic__q,
        r=hex_basic__r,
        s=hex_basic__s,
    ))


@pytest.fixture(scope="function")
def hex_basic__q(request: FixtureRequest) -> int:
    return getattr(request, "param", 1)


@pytest.fixture(scope="function")
def hex_basic__r(request: FixtureRequest) -> int:
    return getattr(request, "param", -1)


@pytest.fixture(scope="function")
def hex_basic__s(request: FixtureRequest) -> int:
    return getattr(request, "param", 0)


@pytest.fixture(scope="function")
def hex_fractional(
    request: FixtureRequest,
    hex_fractional__q: float,
    hex_fractional__r: float,
    hex_fractional__s: float
) -> HexFractional:
    return getattr(request, "param", HexFractional(
        q=hex_fractional__q,
        r=hex_fractional__r,
        s=hex_fractional__s,
    ))


@pytest.fixture(scope="function")
def hex_fractional__q(request: FixtureRequest) -> float:
    return getattr(request, "param", 1.0)


@pytest.fixture(scope="function")
def hex_fractional__r(request: FixtureRequest) -> float:
    return getattr(request, "param", 1.0)


@pytest.fixture(scope="function")
def hex_fractional__s(request: FixtureRequest) -> float:
    return getattr(request, "param", 1.0)

