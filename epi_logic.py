"""PPE post-processing, preserving the legacy Docker person-level schema."""
from __future__ import annotations

import time


def person_regions(person):
    x1, y1, x2, y2 = person
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    return (
        (x1 - int(.10*w), y1, x2 + int(.10*w), y1 + int(.38*h)),
        (x1 - int(.08*w), y1 + int(.20*h), x2 + int(.08*w), y1 + int(.78*h)),
    )


def center_inside(box, region):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = region
    return rx1 <= (x1+x2)/2 <= rx2 and ry1 <= (y1+y2)/2 <= ry2


def build_response(detections: list[dict]) -> dict:
    """Associate positive helmet/vest detections using the Docker heuristics.

    Negative classes (e.g. no_helmet) remain observable in raw_detections but
    are never treated as positive PPE. Like the legacy, associations are not
    exclusive: overlapping people may share an equipment detection.
    """
    people = [d for d in detections if d['label'].strip().lower() == 'person']
    equipment = {
        label: [d for d in detections if d['label'].strip().lower() == label]
        for label in ('helmet', 'vest')
    }
    persons_out = []
    max_conf = 0.0
    for idx, person in enumerate(people):
        # The old implementation rounds xyxy before region association.
        box = [int(round(v)) for v in person['bbox_xyxy']]
        head, torso = person_regions(box)
        selected = {}
        for label, region in (('helmet', head), ('vest', torso)):
            candidates = [d for d in equipment[label]
                          if center_inside([int(round(v)) for v in d['bbox_xyxy']], region)]
            selected[label] = max(candidates, key=lambda d: d['confidence'], default=None)
        helmet, vest = selected['helmet'], selected['vest']
        x1, y1, x2, y2 = box
        persons_out.append({
            'person_id': idx,
            'coordinates': {'x': int((x1+x2)/2), 'y': int((y1+y2)/2),
                            'width': x2-x1, 'height': y2-y1},
            'person_confidence': str(person['confidence']),
            'helmet': helmet is not None,
            'helmet_confidence': str(helmet['confidence']) if helmet else '0.0',
            'vest': vest is not None,
            'vest_confidence': str(vest['confidence']) if vest else '0.0',
            'compliant': helmet is not None and vest is not None,
        })
        max_conf = max(max_conf, person['confidence'],
                       helmet['confidence'] if helmet else 0.0,
                       vest['confidence'] if vest else 0.0)

    non_compliant = any(not p['compliant'] for p in persons_out)
    legacy = {'detections': persons_out, 'non_compliant': non_compliant, 'time': time.time()}
    counts = {}
    for d in detections:
        counts[d['label']] = counts.get(d['label'], 0) + 1
    return {
        **legacy,
        # detection means an actionable EPI alert, not simply the presence of PPE.
        'detection': non_compliant,
        'bbox': [[*d['bbox_xyxy'], d['confidence'], d['label']] for d in detections],
        'detection_count': len(detections),
        'person_count': len(persons_out),
        'non_compliant_count': sum(not p['compliant'] for p in persons_out),
        'counts_by_label': counts,
        'raw_detections': detections,
        'EPIDetection': legacy,
        'EPIMaxConfidence': str(max_conf),
    }
