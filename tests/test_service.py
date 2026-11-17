"""Service tests with synthetic YOLO results (no weight load/download)."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from service import EPIDetection


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = object.__new__(EPIDetection.inner)
        self.service.model = Mock()
        self.image = np.zeros((400, 400, 3), dtype=np.uint8)

    def test_threshold_validation(self):
        for value in (-1, 1.1, float('nan'), float('inf'), None, 'bad'):
            for key in ('confidence_threshold', 'person_threshold', 'helmet_threshold',
                        'vest_threshold', 'iou_threshold'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.service._thresholds({key: value})

    def test_per_class_thresholds_and_original_labels(self):
        result = SimpleNamespace(
            names={6: 'Person', 0: 'helmet', 2: 'vest', 1: 'gloves'},
            boxes=SimpleNamespace(
                xyxy=Tensor([[100, 100, 200, 400], [120, 100, 180, 160],
                             [120, 180, 180, 290], [120, 250, 180, 280]]),
                conf=Tensor([0.4, 0.44, 0.5, 0.24]),
                cls=Tensor([6, 0, 2, 1]),
            ),
        )
        self.service.model.predict.return_value = [result]
        out = self.service._predict_frame(self.image, {})
        self.assertEqual(out['counts_by_label'], {'Person': 1, 'vest': 1})
        self.assertTrue(out['detection'])
        self.assertFalse(out['detections'][0]['helmet'])
        out = self.service._predict_frame(self.image, {'helmet_threshold': 0.4})
        self.assertFalse(out['detection'])
        self.assertEqual(self.service.model.predict.call_args.kwargs['conf'], 0.25)

    def test_no_boxes(self):
        self.service.model.predict.return_value = [SimpleNamespace(boxes=None)]
        out = self.service._predict_frame(self.image, {})
        self.assertEqual(out['bbox'], [])
        self.assertFalse(out['non_compliant'])


if __name__ == '__main__':
    unittest.main()
