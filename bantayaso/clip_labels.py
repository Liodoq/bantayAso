"""Owner-written clip descriptions. These do not train models or change the video."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def labels_path(clip: Path) -> Path:
    clip = Path(clip)
    return clip.with_name(clip.name + '.labels.json')


def load_labels(clip: Path) -> dict:
    path = labels_path(clip)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('kind') != 'bantayaso.clip_labels' or data.get('version') != 1:
        raise ValueError('This clip has an unrecognized label file. It has not been changed.')
    if data.get('clip') != Path(clip).name:
        raise ValueError('The existing labels refer to another clip. They have not been changed.')
    for key in ('dogs', 'behavior', 'notes'):
        if not isinstance(data.get(key, ''), str):
            raise ValueError('The clip label fields are invalid. They have not been changed.')
    return data


def save_labels(clip: Path, dogs: str, behavior: str, notes: str) -> Path:
    clip = Path(clip)
    if not clip.is_file():
        raise ValueError('The video file cannot be found. Your labels have not been saved.')
    dogs, behavior, notes = dogs.strip(), behavior.strip(), notes.strip()
    if not behavior:
        raise ValueError('Enter a behavior, or choose Mixed / unclear.')
    if not dogs and behavior.casefold() != 'no dog in view':
        raise ValueError('Enter the dog name(s), or use Unknown dog.')
    data = load_labels(clip)
    data.update(kind='bantayaso.clip_labels', version=1, clip=clip.name, dogs=dogs,
                behavior=behavior, notes=notes, annotation_scope='clip_summary',
                source='owner', updated_at=datetime.now(timezone.utc).isoformat())
    dest = labels_path(clip)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=dest.parent,
                                         suffix='.labels.tmp', delete=False) as f:
            tmp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(tmp, dest)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return dest
