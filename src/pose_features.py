# src/pose_features.py
# Hand -> fixed-length feature vector for static pose classification.
import numpy as np
import leap

N_FEATURES = 62   # 5 fingers x 4 joints x 3 + pinch + grab

def _v(p):
    return np.array([p.x, p.y, p.z], dtype=np.float32)

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v

def pose_features(hand):
    palm = _v(hand.palm.position)
    normal = _unit(_v(hand.palm.normal))
    direction = _unit(_v(hand.palm.direction))
    side = _unit(np.cross(direction, normal))
    R = np.stack([side, normal, direction])        # world -> palm frame

    # hand size: palm centre to middle finger knuckle
    knuckle = _v(hand.digits[2].metacarpal.next_joint)
    scale = max(np.linalg.norm(knuckle - palm), 1e-3)

    mirror = -1.0 if hand.type == leap.HandType.Left else 1.0
    feats = []
    for d in hand.digits:
        for b in (d.metacarpal, d.proximal, d.intermediate, d.distal):
            local = R @ (_v(b.next_joint) - palm) / scale
            local[0] *= mirror
            feats.extend(local.tolist())
    feats.append(float(hand.pinch_strength))
    feats.append(float(hand.grab_strength))
    return feats
