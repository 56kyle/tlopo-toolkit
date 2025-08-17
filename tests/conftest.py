"""Fixtures used in all tests."""
import pytest
from _pytest.fixtures import FixtureRequest

from tlopo_toolkit.geometry import Hex
from tlopo_toolkit.geometry import HexFractional
from tlopo_toolkit.geometry import _Hex


pytest_plugins: list[str] = [
    "tests.plugins.geometry"
]


