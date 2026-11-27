"""Select a court calibration by matching the visible camera background."""

import base64
import json
from pathlib import Path

import cv2
import numpy as np

def camera_signature(frame: np.ndarray) -> np.ndarray:
    """Summarize the wall and ceiling above the court, where players seldom appear."""
    height, width = frame.shape[:2]
    section = frame[int(height * .07):int(height * .42), int(width * .04):int(width * .96)]
    gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 12), interpolation=cv2.INTER_AREA)


def find_court_boundary(video: Path, profiles: Path) -> Path | None:
    """Return a calibration only for an unambiguous, close camera match."""
    candidates = sorted(profiles.glob("*.camera.json"))
    if not candidates:
        return None
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or min(frame.shape[:2]) < 32:
        return None
    signature = camera_signature(frame)
    ratio = frame.shape[1] / frame.shape[0]
    matches = []
    for path in candidates:
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            aspect_ratio = float(profile["aspect_ratio"])
            boundary = profiles / profile["boundary"]
            encoded_signatures = [profile["signature"], *profile.get("additional_signatures", ())]
            references = [np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8).reshape(signature.shape)
                          for encoded in encoded_signatures]
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if aspect_ratio <= 0 or abs(ratio / aspect_ratio - 1) > .025:
            continue
        score = max((float(cv2.matchTemplate(signature, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
                     for reference in references if np.std(reference) >= 2), default=0)
        if score >= .93:
            matches.append((score, boundary))
    matches.sort(reverse=True)
    if not matches or len(matches) > 1 and matches[0][0] - matches[1][0] < .03:
        return None
    return matches[0][1]
