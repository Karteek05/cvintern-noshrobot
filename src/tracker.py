"""
Minimal multi-object tracker: greedy nearest-centroid matching frame-to-frame,
with short-term memory so a track survives a few missed detections (e.g. an
egg briefly hidden behind a hand, or a frame where the classical segmenter
just didn't find it). No Kalman filter / motion model -- with only two slow,
mostly-stationary eggs, "match to nearest centroid, predict as last-known
position while lost" is enough, and it's much easier to explain than a full
SORT implementation.
"""
from dataclasses import dataclass, field


@dataclass
class Track:
    track_id: int
    centroid: tuple
    bbox: tuple  # (x0, y0, x1, y1)
    age: int = 0
    frames_since_update: int = 0
    hits: int = 0
    history: list = field(default_factory=list)  # list of (frame_idx, centroid) while visible


class CentroidTracker:
    def __init__(self, max_tracks=None, max_distance=90, max_missed=20):
        """
        max_tracks: cap on simultaneously active tracks (None = unbounded).
            Set to a known object count (e.g. 2 eggs) to reject spurious
            detections -- e.g. a glossy-table reflection -- from spawning a
            third track once both real eggs are already being tracked.
        max_distance: gating radius (px) for matching a detection to a track.
        max_missed: frames a track may go unmatched before being dropped.
        """
        self.max_tracks = max_tracks
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks = []
        self._next_id = 1

    def update(self, detections, frame_idx):
        """detections: list of (centroid(x,y), bbox(x0,y0,x1,y1)). Returns the
        current list of active Track objects (including ones predicted
        through a brief occlusion)."""
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        pairs = []
        for ti in unmatched_tracks:
            for di in unmatched_dets:
                d = _dist(self.tracks[ti].centroid, detections[di][0])
                if d <= self.max_distance:
                    pairs.append((d, ti, di))
        pairs.sort(key=lambda p: p[0])

        used_tracks, used_dets = set(), set()
        for d, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            t = self.tracks[ti]
            centroid, bbox = detections[di]
            t.centroid = centroid
            t.bbox = bbox
            t.frames_since_update = 0
            t.hits += 1
            t.history.append((frame_idx, centroid))

        for ti in range(len(self.tracks)):
            if ti not in used_tracks:
                self.tracks[ti].frames_since_update += 1
            self.tracks[ti].age += 1

        self.tracks = [t for t in self.tracks if t.frames_since_update <= self.max_missed]

        for di in range(len(detections)):
            if di in used_dets:
                continue
            if self.max_tracks is not None and len(self.tracks) >= self.max_tracks:
                continue  # at capacity -- treat as noise (e.g. a reflection)
            centroid, bbox = detections[di]
            t = Track(track_id=self._next_id, centroid=centroid, bbox=bbox, hits=1)
            t.history.append((frame_idx, centroid))
            self._next_id += 1
            self.tracks.append(t)

        return self.tracks


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
