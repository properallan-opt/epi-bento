from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import bentoml
import numpy as np
from ultralytics import YOLO

from epi_logic import build_response
from saci_yolo import BaseYOLOService

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv('EPI_MODEL_PATH', 'model_weights/ppe-v2-100epochs.pt'))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH
IMAGE_SIZE = int(os.getenv('EPI_IMAGE_SIZE', '640'))
DEVICE = os.getenv('EPI_DEVICE', 'cpu')
CONFIDENCE_THRESHOLD = float(os.getenv('EPI_CONFIDENCE_THRESHOLD', '0.25'))
PERSON_THRESHOLD = float(os.getenv('EPI_PERSON_THRESHOLD', '0.35'))
HELMET_THRESHOLD = float(os.getenv('EPI_HELMET_THRESHOLD', '0.45'))
VEST_THRESHOLD = float(os.getenv('EPI_VEST_THRESHOLD', '0.45'))
IOU_THRESHOLD = float(os.getenv('EPI_IOU_THRESHOLD', '0.70'))


@bentoml.service(workers=1)
class EPIDetection(BaseYOLOService):
    """Single-model PPE detector with SACI transports and legacy person output."""

    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f'EPI weights not found: {MODEL_PATH}')
        self.model = YOLO(str(MODEL_PATH), task='detect')
        labels = {str(label).strip().lower() for label in self.model.names.values()}
        if not {'person', 'helmet', 'vest'} <= labels:
            raise ValueError('EPI weights must contain Person, helmet and vest classes.')

    @staticmethod
    def _thresholds(request: Mapping[str, Any]) -> dict[str, float]:
        defaults = {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'person_threshold': PERSON_THRESHOLD,
            'helmet_threshold': HELMET_THRESHOLD,
            'vest_threshold': VEST_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD,
        }
        values = {}
        for key, default in defaults.items():
            try:
                value = float(request.get(key, default))
            except (TypeError, ValueError) as error:
                raise ValueError(f'{key} must be between 0 and 1.') from error
            if not 0 <= value <= 1:
                raise ValueError(f'{key} must be between 0 and 1.')
            values[key] = value
        return values

    def _predict_frame(self, image: np.ndarray, request: Mapping[str, Any]) -> dict:
        thresholds = self._thresholds(request)
        by_label = {label: thresholds[f'{label}_threshold']
                    for label in ('person', 'helmet', 'vest')}
        result = self.model.predict(
            source=image, imgsz=IMAGE_SIZE,
            conf=min(thresholds['confidence_threshold'], *by_label.values()),
            iou=thresholds['iou_threshold'], device=DEVICE, verbose=False,
        )[0]
        detections = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                class_id, confidence = int(class_id), float(confidence)
                label = str(result.names[class_id])
                threshold = by_label.get(label.strip().lower(), thresholds['confidence_threshold'])
                if confidence < threshold:
                    continue
                detections.append({
                    'class_id': class_id, 'label': label,
                    'confidence': confidence, 'bbox_xyxy': [float(v) for v in xyxy],
                })
        return build_response(detections)

    @staticmethod
    def _request(confidence_threshold, iou_threshold, person_threshold,
                 helmet_threshold, vest_threshold):
        return {
            'confidence_threshold': confidence_threshold, 'iou_threshold': iou_threshold,
            'person_threshold': person_threshold, 'helmet_threshold': helmet_threshold,
            'vest_threshold': vest_threshold,
        }

    @bentoml.api
    def recognize(
        self, image: Path,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        person_threshold: float = PERSON_THRESHOLD,
        helmet_threshold: float = HELMET_THRESHOLD,
        vest_threshold: float = VEST_THRESHOLD,
    ):
        return self._from_path(image, self._request(
            confidence_threshold, iou_threshold, person_threshold, helmet_threshold, vest_threshold))

    @bentoml.api
    def recognize_array(
        self, image: np.ndarray,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        person_threshold: float = PERSON_THRESHOLD,
        helmet_threshold: float = HELMET_THRESHOLD,
        vest_threshold: float = VEST_THRESHOLD,
    ):
        return self._from_array(image, self._request(
            confidence_threshold, iou_threshold, person_threshold, helmet_threshold, vest_threshold))

    @bentoml.api
    def recognize_path(self, request: dict, /):
        try:
            image_path = Path(request['image_path'])
        except (KeyError, TypeError) as error:
            raise ValueError('image_path is required.') from error
        return self._from_path(image_path, request)

    @bentoml.api
    def recognize_base64(self, request: dict, /):
        return self._from_base64(request)

    @bentoml.api
    def recognize_raw(self, packet: bytes, /):
        return self._from_raw(packet)

    @bentoml.api
    def recognize_shm(self, request: dict, /):
        return self._from_shm(request)
