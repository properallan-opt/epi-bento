from __future__ import annotations

import base64
import binascii
import json
import struct
import mmap
import os
import _posixshmem
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MAX_IMAGE_BYTES = 10 * 1024 * 1024 * 1024
RAW_HEADER_LENGTH_BYTES = 4


class BaseYOLOService:
    """Reusable frame transport/decoding helpers for BentoML model services.

    Subclasses only need to implement ``_predict_frame``. Model-specific
    metadata can be carried in the request dictionary and interpreted by the
    subclass.
    """

    max_image_bytes = MAX_IMAGE_BYTES
    raw_header_length_bytes = RAW_HEADER_LENGTH_BYTES

    def _predict_frame(
        self,
        image: np.ndarray,
        request: Mapping[str, Any],
    ) -> dict:
        raise NotImplementedError

    def _validate_bgr_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Image must have shape (height, width, 3). "
                f"Received: {image.shape}"
            )

        if image.dtype != np.uint8:
            raise ValueError(
                "Image dtype must be uint8. "
                f"Received: {image.dtype}"
            )

        if image.nbytes > self.max_image_bytes:
            raise ValueError(
                "Image exceeds the allowed size: "
                f"{image.nbytes} bytes"
            )

        return np.ascontiguousarray(image)

    def _finish_request(
        self,
        image: np.ndarray,
        request: Mapping[str, Any] | None = None,
    ) -> dict:
        image = self._validate_bgr_image(image)
        return self._predict_frame(image, request or {})

    def _from_path(
        self,
        image_path: str | Path,
        request: Mapping[str, Any] | None = None,
    ) -> dict:
        import cv2
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Could not decode the received image."
            )

        return self._finish_request(image, request)

    def _from_array(
        self,
        image: np.ndarray,
        request: Mapping[str, Any] | None = None,
    ) -> dict:
        image = np.asarray(
            image,
            dtype=np.uint8,
            order="C",
        )

        return self._finish_request(image, request)

    def _from_base64(
        self,
        request: Mapping[str, Any],
    ) -> dict:
        import cv2
        try:
            encoded_image = base64.b64decode(
                request["image_base64"],
                validate=True,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            binascii.Error,
        ) as error:
            raise ValueError(
                "Missing or invalid image_base64 field."
            ) from error

        image = cv2.imdecode(
            np.frombuffer(encoded_image, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Could not decode the Base64 image."
            )

        return self._finish_request(image, request)

    def _from_raw(self, packet: bytes) -> dict:
        if len(packet) < self.raw_header_length_bytes:
            raise ValueError("Incomplete raw packet.")

        header_size = struct.unpack(
            "!I",
            packet[: self.raw_header_length_bytes],
        )[0]

        header_start = self.raw_header_length_bytes
        header_end = header_start + header_size

        if header_end > len(packet):
            raise ValueError(
                "Declared header exceeds packet size."
            )

        try:
            metadata = json.loads(
                packet[header_start:header_end].decode("utf-8")
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                "Invalid raw JSON header."
            ) from error

        if not isinstance(metadata, dict):
            raise ValueError(
                "Raw JSON header must be an object."
            )

        try:
            shape = tuple(int(value) for value in metadata["shape"])
            dtype = np.dtype(metadata["dtype"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Raw header must contain valid shape and dtype."
            ) from error

        self._validate_shape_dtype(shape, dtype, source="raw")

        element_count = int(np.prod(shape))
        expected_size = element_count * dtype.itemsize
        raw_size = len(packet) - header_end

        if expected_size != raw_size:
            raise ValueError(
                "Pixel size is incompatible with shape/dtype: "
                f"expected={expected_size}, received={raw_size}"
            )

        if expected_size > self.max_image_bytes:
            raise ValueError(
                "Raw image exceeds the allowed size: "
                f"{expected_size} bytes"
            )

        image = np.frombuffer(
            packet,
            dtype=dtype,
            count=element_count,
            offset=header_end,
        ).reshape(shape)

        return self._finish_request(image, metadata)

    def _from_shm(
        self,
        request: Mapping[str, Any],
    ) -> dict:
        try:
            descriptor = request["image"]
            name = descriptor["name"]
            shape = tuple(int(value) for value in descriptor["shape"])
            dtype = np.dtype(descriptor["dtype"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Invalid SHM payload. Expected: "
                "{'image': {'name', 'shape', 'dtype'}, ...}"
            ) from error

        self._validate_shape_dtype(shape, dtype, source="SHM")

        expected_size = shape[0] * shape[1] * shape[2] * dtype.itemsize

        if expected_size > self.max_image_bytes:
            raise ValueError(
                "SHM image exceeds the allowed size: "
                f"{expected_size} bytes"
            )

        if (not isinstance(name, str) or not name or "\x00" in name
                or not name.lstrip("/") or "/" in name.lstrip("/")):
            raise ValueError("Invalid SHM name")

        # Attach only; the producer owns creation and unlinking.
        # Bypass SharedMemory to avoid resource_tracker registration.
        fd = _posixshmem.shm_open("/" + name.lstrip("/"), os.O_RDONLY, mode=0o600)
        try:
            available_size = os.fstat(fd).st_size
            if expected_size > available_size:
                raise ValueError(
                    "Shape/dtype exceed the shared-memory segment: "
                    f"expected={expected_size}, available={available_size}"
                )
            mapping = mmap.mmap(fd, expected_size, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)

        try:
            image_view = np.ndarray(shape=shape, dtype=dtype, buffer=mapping)
            try:
                image = image_view.copy()
            finally:
                del image_view
        finally:
            mapping.close()

        return self._finish_request(image, request)

    @staticmethod
    def _validate_shape_dtype(
        shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        source: str,
    ) -> None:
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(
                f"Invalid BGR shape for {source}: {shape}"
            )

        if any(value <= 0 for value in shape):
            raise ValueError(
                f"Invalid non-positive dimension for {source}: {shape}"
            )

        if dtype != np.uint8:
            raise ValueError(
                f"Only uint8 is accepted for {source}: {dtype}"
            )