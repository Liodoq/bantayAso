"""Bounded inference check on existing clips; no camera, downloads requested, or training writes.

python scripts/check_inference.py data/clips/rec_20261009_175137.mp4 --device cuda
This is a load/inference check, NOT a labeled accuracy benchmark or live FPS test.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seconds', type=float, nargs='+', default=[0, 1, 2])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not args.clip.is_file():
        parser.error('Use an existing recorded clip.')
    import cv2
    import numpy as np
    from ultralytics.utils import LOGGER
    from bantayaso import config
    from bantayaso.detect_dog import DogDetector
    from bantayaso.detect_hazards import HazardDetector
    from bantayaso.actions import ActionClassifier
    LOGGER.setLevel(logging.ERROR)
    cfg = config.load()
    for name in [cfg['models']['dog_detector'], cfg['models']['hazard_detector'],
                 'clip/ViT-B-32.pt', 'mobileclip_blt.ts']:
        if not (config.MODELS_DIR/name).is_file():
            parser.error(f'Missing cached weights: {name}; this check does not install them.')
    print('Loading cached detection and action models...', flush=True)
    dogs = DogDetector(config.MODELS_DIR/cfg['models']['dog_detector'], device=args.device,
                       conf=cfg.get('detect', {}).get('dog_conf', .18), imgsz=960)
    hazards = HazardDetector(config.MODELS_DIR/cfg['models']['hazard_detector'], cfg['hazards'],
                             device=args.device, conf=.25, imgsz=640)
    actions = ActionClassifier(cfg['actions'], config.MODELS_DIR, device=args.device,
                               model_name=cfg['models'].get('clip_openai_name', 'ViT-B/32'))
    cap = cv2.VideoCapture(str(args.clip))
    rows = []
    try:
        for second in args.seconds:
            cap.set(cv2.CAP_PROP_POS_MSEC, second*1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read clip at {second} seconds')
            frame = cv2.resize(frame, (1280, 720))
            ds, hs = dogs(frame), hazards(frame)
            aa = actions(frame, ds, collect=False)
            row = {'second': second, 'dogs': len(ds), 'hazards': [h.name for h in hs],
                   'raw_actions': [{'track_id': tid, 'label': a.label,
                                    'score': round(a.conf, 3), 'eat_score': round(.5*(a.chew+a.mouth), 3)}
                                   for tid, a in aa.items()]}
            for a in aa.values():
                if not np.isfinite([a.conf, a.chew, a.mouth]).all():
                    raise RuntimeError('Non-finite model output')
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        cap.release()
    report = {'clip': str(args.clip), 'device': args.device, 'check': 'load and sampled inference only',
              'saved_action_classes': len(actions.examples), 'rows': rows}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('INFERENCE CHECK PASS; accuracy and live performance not measured.', flush=True)


if __name__ == '__main__':
    main()
