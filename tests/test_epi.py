import base64
import json
from multiprocessing import shared_memory
from pathlib import Path
import struct
import tempfile
import unittest

import cv2
import numpy as np

from epi_logic import build_response
from saci_yolo import BaseYOLOService


def det(label, box, confidence=0.9):
    return {'class_id': 0, 'label': label, 'bbox_xyxy': box, 'confidence': confidence}


class LogicTests(unittest.TestCase):
    person = det('Person', [100, 100, 200, 400], 0.8)
    helmet = det('helmet', [120, 100, 180, 160])
    vest = det('vest', [120, 180, 180, 290])

    def test_compliant_legacy_fields(self):
        out = build_response([self.person, self.helmet, self.vest])
        self.assertFalse(out['detection'])
        self.assertEqual(out['person_count'], 1)
        person = out['detections'][0]
        self.assertTrue(person['compliant'])
        self.assertEqual(person['coordinates'], {'x': 150, 'y': 250, 'width': 100, 'height': 300})
        self.assertIsInstance(person['helmet_confidence'], str)
        self.assertEqual(out['EPIMaxConfidence'], '0.9')
        self.assertEqual(out['EPIDetection']['detections'], out['detections'])
        json.dumps(out, allow_nan=False)

    def test_missing_ppe_alert(self):
        for equipment in ([], [self.helmet], [self.vest]):
            with self.subTest(equipment=equipment):
                out = build_response([self.person, *equipment])
                self.assertTrue(out['detection'])
                self.assertTrue(out['non_compliant'])
                self.assertEqual(out['non_compliant_count'], 1)

    def test_empty_and_equipment_without_people(self):
        for detections in ([], [self.helmet, self.vest]):
            out = build_response(detections)
            self.assertFalse(out['detection'])
            self.assertEqual(out['detections'], [])
            self.assertEqual(out['EPIMaxConfidence'], '0.0')

    def test_negative_classes_are_not_positive_equipment(self):
        out = build_response([self.person, det('no_helmet', [120, 100, 180, 160]),
                              det('gloves', [120, 180, 180, 290])])
        self.assertFalse(out['detections'][0]['helmet'])
        self.assertEqual(out['detection_count'], 3)

    def test_equipment_outside_regions(self):
        out = build_response([self.person, det('helmet', [500, 500, 550, 550]), self.vest])
        self.assertTrue(out['detection'])
        self.assertFalse(out['detections'][0]['helmet'])

    def test_multiple_people(self):
        out = build_response([self.person, self.helmet, self.vest,
                              det('person', [500, 100, 600, 400])])
        self.assertEqual(out['person_count'], 2)
        self.assertEqual(out['non_compliant_count'], 1)
        self.assertEqual([p['person_id'] for p in out['detections']], [0, 1])


class EchoService(BaseYOLOService):
    def _predict_frame(self, image, request):
        return {'shape': list(image.shape), 'sum': int(image.sum()), 'tag': request.get('tag')}


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.service = EchoService()
        self.image = np.full((8, 12, 3), 42, dtype=np.uint8)
        self.expected = {'shape': [8, 12, 3], 'sum': int(self.image.sum()), 'tag': 'test'}
        self.request = {'tag': 'test'}

    def test_array_path_base64_raw_shm(self):
        self.assertEqual(self.service._from_array(self.image, self.request), self.expected)
        encoded = cv2.imencode('.png', self.image)[1].tobytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.png'
            path.write_bytes(encoded)
            self.assertEqual(self.service._from_path(path, self.request), self.expected)
        self.assertEqual(self.service._from_base64({**self.request,
            'image_base64': base64.b64encode(encoded).decode()}), self.expected)
        header = json.dumps({**self.request, 'shape': list(self.image.shape), 'dtype': 'uint8'}).encode()
        packet = struct.pack('!I', len(header)) + header + self.image.tobytes()
        self.assertEqual(self.service._from_raw(packet), self.expected)
        shm = shared_memory.SharedMemory(create=True, size=self.image.nbytes)
        try:
            shm.buf[:] = self.image.tobytes()
            request = {**self.request, 'image': {'name': shm.name,
                       'shape': list(self.image.shape), 'dtype': 'uint8'}}
            self.assertEqual(self.service._from_shm(request), self.expected)
            # The consumer must not unlink producer-owned shared memory.
            attached = shared_memory.SharedMemory(name=shm.name)
            attached.close()
        finally:
            shm.close()
            shm.unlink()

    def test_invalid_packets(self):
        for packet in (b'', struct.pack('!I', 100), struct.pack('!I', 2) + b'[]'):
            with self.subTest(packet=packet), self.assertRaises(ValueError):
                self.service._from_raw(packet)
        with self.assertRaises(ValueError):
            self.service._from_base64({'image_base64': 'not base64'})
        with self.assertRaises(ValueError):
            self.service._from_shm({'image': {'name': 'bad/name', 'shape': [1, 1, 3], 'dtype': 'uint8'}})


if __name__ == '__main__':
    unittest.main()
