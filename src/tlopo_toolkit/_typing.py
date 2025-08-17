"""Module containing custom typing used throughout the tlopo-toolkit package."""

from typing import Annotated
from typing import Any
from typing import TypeAlias

import numpy as np
from pydantic import AfterValidator


def validate_gray(color: np.ndarray[tuple[int, ...], np.dtype[Any]]) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    """Validate grayscale array"""
    if color.shape[-1] != 1 or color.dtype != np.uint8:
        raise ValueError(f"Grayscale must be uint8 with 1 channel, got shape {color.shape}, dtype {color.dtype}")
    return color


def validate_rgb(color: np.ndarray[tuple[int, ...], np.dtype[Any]]) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    """Validate RGB array"""
    if color.shape[-1] not in [3, 4] or color.dtype != np.uint8:
        raise ValueError(f"RGB must be uint8 with 3-4 channels, got shape {color.shape}, dtype {color.dtype}")
    return color


def validate_bgr(color: np.ndarray[tuple[int, ...], np.dtype[Any]]) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    """Validate BGR array"""
    if color.shape[-1] not in [3, 4] or color.dtype != np.uint8:
        raise ValueError(f"BGR must be uint8 with 3-4 channels, got shape {color.shape}, dtype {color.dtype}")
    return color


def validate_hsv(
    color: np.ndarray[tuple[int, ...], np.dtype[Any]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Validate HSV array"""
    if color.shape[-1] != 3 or color.dtype != np.float32:
        raise ValueError(f"HSV must be float32 with 3 channels, got shape {color.shape}, dtype {color.dtype}")
    return color


def validate_lab(
    color: np.ndarray[tuple[int, ...], np.dtype[Any]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    """Validate LAB array"""
    if color.shape[-1] != 3 or color.dtype != np.float32:
        raise ValueError(f"LAB must be float32 with 3 channels, got shape {color.shape}, dtype {color.dtype}")
    return color


def validate_rgba(color: np.ndarray[tuple[int, ...], np.dtype[Any]]) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    """Validate RGBA array"""
    if color.shape[-1] != 4 or color.dtype != np.uint8:
        raise ValueError(f"RGBA must be uint8 with 4 channels, got shape {color.shape}, dtype {color.dtype}")
    return color


def validate_bgra(color: np.ndarray[tuple[int, ...], np.dtype[Any]]) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    """Validate BGRA array"""
    if color.shape[-1] != 4 or color.dtype != np.uint8:
        raise ValueError(f"BGRA must be uint8 with 4 channels, got shape {color.shape}, dtype {color.dtype}")
    return color


GRAY: TypeAlias = Annotated[np.ndarray[tuple[1], np.dtype[np.uint8]], AfterValidator(validate_gray)]
RGB: TypeAlias = Annotated[np.ndarray[tuple[3], np.dtype[np.uint8]], AfterValidator(validate_rgb)]
BGR: TypeAlias = Annotated[np.ndarray[tuple[3], np.dtype[np.uint8]], AfterValidator(validate_bgr)]
HSV: TypeAlias = Annotated[np.ndarray[tuple[3], np.dtype[np.float32]], AfterValidator(validate_hsv)]
LAB: TypeAlias = Annotated[np.ndarray[tuple[3], np.dtype[np.float32]], AfterValidator(validate_lab)]
RGBA: TypeAlias = Annotated[np.ndarray[tuple[4], np.dtype[np.uint8]], AfterValidator(validate_rgba)]
BGRA: TypeAlias = Annotated[np.ndarray[tuple[4], np.dtype[np.uint8]], AfterValidator(validate_bgra)]

ImgGRAY: TypeAlias = Annotated[np.ndarray[tuple[int, int, 1], np.dtype[np.uint8]], AfterValidator(validate_gray)]
ImgRGB: TypeAlias = Annotated[np.ndarray[tuple[int, int, 3], np.dtype[np.uint8]], AfterValidator(validate_rgb)]
ImgBGR: TypeAlias = Annotated[np.ndarray[tuple[int, int, 3], np.dtype[np.uint8]], AfterValidator(validate_bgr)]
ImgHSV: TypeAlias = Annotated[np.ndarray[tuple[int, int, 3], np.dtype[np.float32]], AfterValidator(validate_hsv)]
ImgLAB: TypeAlias = Annotated[np.ndarray[tuple[int, int, 3], np.dtype[np.float32]], AfterValidator(validate_lab)]
ImgRGBA: TypeAlias = Annotated[np.ndarray[tuple[int, int, 4], np.dtype[np.uint8]], AfterValidator(validate_rgba)]
ImgBGRA: TypeAlias = Annotated[np.ndarray[tuple[int, int, 4], np.dtype[np.uint8]], AfterValidator(validate_bgra)]
