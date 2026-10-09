"""Conservative, deterministic check-ins. No LLM and no schedules."""
class Companion:
    def __init__(self, dwell=60.0, cooldown=900.0):
        self.dwell, self.cooldown = dwell, cooldown
        self.key = None
        self.since = 0.0
        self.last_frame = None
        self.announced = False
        self.last_spoken = float('-inf')

    def update(self, now, observations, blocked=False):
        # observations: (track_id, display_name, action, eligible). Held detections never qualify.
        key = tuple(sorted((tid, name) for tid, name, action, valid in observations))
        calm = bool(observations) and all(valid and action in ('sleeping', 'lying down')
                                         for tid, name, action, valid in observations)
        if self.last_frame is None or now - self.last_frame > 2 or not calm or key != self.key:
            self.key, self.since, self.announced = key if calm else None, now, False
        self.last_frame = now
        if not calm or blocked or self.announced or now - self.since < self.dwell or now - self.last_spoken < self.cooldown:
            return None
        names = [name for _, name in key]
        subject = ' and '.join(names) if all(names) and len(set(names)) == len(names) else ('The dog in view' if len(names) == 1 else 'The dogs in view')
        return subject + (' appears to be resting.' if len(names) == 1 else ' appear to be resting.')

    def mark_spoken(self, now):
        self.announced = True
        self.last_spoken = now
