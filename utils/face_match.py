#!/usr/bin/env python3
"""
Lightweight multi-frame face matcher.

Given RetinaFace detections from N images shot ~180° around a subject,
finds faces appearing consistently across frames and assigns global IDs.

Strategy: chain pairwise Hungarian matches across consecutive frames,
then build connected components.  Cost combines:
  - bbox area + aspect ratio (pose-stable)
  - bbox centre position  (strong spatial prior for orbiting camera)
  - landmark geometry     (auxiliary, weighted down when landmarks look noisy)
  - [optional] image patch features  (colour moments or custom embeddings)

Usage:
    from face_matcher import FaceMatcher

    matcher = FaceMatcher(strategy="combined", min_frames=3)
    matches = matcher.match(bboxes_list, confs_list, landmarks_list)

    # With image-based disambiguation:
    img_feats = extract_patch_features(images, bboxes_list)
    matches = matcher.match(..., image_features_list=img_feats)
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

# ── Hungarian (scipy → greedy fallback) ──────────────────────────────────
try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _hungarian_match(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Min-cost one-to-one assignment.  Falls back to greedy if scipy is missing."""
    if _HAS_SCIPY:
        r, c = _hungarian(cost)
        return list(zip(r.tolist(), c.tolist()))

    matches = []
    used_c = set()
    order = np.dstack(np.unravel_index(np.argsort(cost.ravel()), cost.shape))[0]
    for ri, ci in order:
        if ri not in {m[0] for m in matches} and ci not in used_c:
            matches.append((int(ri), int(ci)))
            used_c.add(ci)
    return matches


# ── Per-face descriptors ─────────────────────────────────────────────────

def _landmark_descriptor(ldm5: np.ndarray) -> np.ndarray:
    """5-pt → 4-D scale-invariant descriptor.

    ldm5: (5,2)  [left-eye, right-eye, nose, left-mouth, right-mouth]
    Returns [nose_x, nose_y, mouth_x, mouth_y] / eye_distance.
    """
    le, re, nose, lm, rm = ldm5
    ec = (le + re) * 0.5
    mc = (lm + rm) * 0.5
    ed = float(np.linalg.norm(re - le))
    if ed < 1e-6:
        return np.zeros(4, dtype=np.float32)
    return np.concatenate([(nose - ec) / ed, (mc - ec) / ed]).astype(np.float32)


def _landmark_quality(ldm5: np.ndarray, bbox: List[float]) -> float:
    """Heuristic landmark quality ∈ [0, 1].

    Returns 0 when eye-distance is implausibly small relative to bbox width
    (RetinaFace often produces degenerate landmarks on profile views).
    """
    le, re = ldm5[0], ldm5[1]
    ed = float(np.linalg.norm(re - le))
    bw = max(float(bbox[2]) - float(bbox[0]), 1.0)
    ratio = ed / bw
    if ratio >= 0.15:
        return 1.0
    return ratio / 0.15


def _bbox_descriptor(bbox: List[float], ref_scale: float = 1.0) -> np.ndarray:
    """bbox [x1,y1,x2,y2] → [log_area, aspect, cx/ref, cy/ref].

    ref_scale is typically the global median face width, so position features
    are ≈ O(1) and comparable to log_area / aspect.
    """
    w = max(float(bbox[2]) - float(bbox[0]), 1.0)
    h = max(float(bbox[3]) - float(bbox[1]), 1.0)
    cx = float(bbox[0]) + w * 0.5
    cy = float(bbox[1]) + h * 0.5
    s = max(ref_scale, 1.0)
    return np.array([np.log(w * h), w / h, cx / s, cy / s], dtype=np.float32)


# ── Lightweight image-patch features ─────────────────────────────────────

def extract_patch_features(
    images: List[np.ndarray],
    bboxes_list: List[List[List[float]]],
    landmarks_list: Optional[List[List[np.ndarray]]] = None,
    colour_space: str = "lab",
    patch_size: int = 48,
    crop_mode: str = "inner",
    inner_ratio: float = 0.55,
) -> List[List[np.ndarray]]:
    """Extract lightweight colour descriptors from face patches.

    For each detection crops a region from its image, resizes to
    patch_size × patch_size, and computes [mean_L, mean_A, mean_B, std_L,
    std_A, std_B] (6-D) in LAB space (or RGB).

    Crop modes (to avoid hair/background that varies with viewpoint):
      'inner'     — centre {inner_ratio} of bbox (default 55% → mostly skin)
      'landmark'  — bounding box of the 5 landmarks + margin (tightest)
      'full'      — entire bbox (legacy; bad for cross-pose matching)

    Parameters
    ----------
    images : list of np.ndarray
        One image per frame, shape (H, W, 3) uint8 RGB.
    bboxes_list : same structure as match()
    landmarks_list : required for crop_mode='landmark', optional otherwise
    colour_space : 'lab' (default, perceptually uniform) or 'rgb'
    patch_size : resize face patch to this square size before stats
    crop_mode : 'inner' | 'landmark' | 'full'
    inner_ratio : fraction of bbox to keep in 'inner' mode (0.55 = centre 55%)

    Returns
    -------
    image_features_list : list of list of np.ndarray
        Same nesting as bboxes_list; each leaf is a (6,) float32 descriptor.
    """
    assert crop_mode in ("inner", "landmark", "full")
    if crop_mode == "landmark" and landmarks_list is None:
        raise ValueError("crop_mode='landmark' requires landmarks_list")

    result = []
    for frame_idx, frame_bboxes in enumerate(bboxes_list):
        img = images[frame_idx]
        h_img, w_img = img.shape[:2]
        frame_feats = []
        for li, bbox in enumerate(frame_bboxes):
            ldm = landmarks_list[frame_idx][li] if landmarks_list else None
            feat = _patch_features_one(
                img, bbox, colour_space, patch_size,
                h_img, w_img, crop_mode, inner_ratio, ldm,
            )
            frame_feats.append(feat)
        result.append(frame_feats)
    return result


def _patch_features_one(
    img: np.ndarray,
    bbox: List[float],
    colour_space: str,
    patch_size: int,
    h_img: int,
    w_img: int,
    crop_mode: str = "inner",
    inner_ratio: float = 0.55,
    ldm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract 6-D colour descriptor from one face patch."""
    # Determine crop rectangle
    if crop_mode == "landmark" and ldm is not None:
        # Use landmark bounding box + 15% margin
        lx1, ly1 = ldm.min(axis=0)
        lx2, ly2 = ldm.max(axis=0)
        lm_w, lm_h = lx2 - lx1, ly2 - ly1
        margin = max(lm_w, lm_h) * 0.15
        x1 = max(0, int(lx1 - margin))
        y1 = max(0, int(ly1 - margin))
        x2 = min(w_img, int(lx2 + margin))
        y2 = min(h_img, int(ly2 + margin))
    elif crop_mode == "inner":
        # Centre inner_ratio of bbox (e.g. 55% → mostly face skin, less hair)
        bw = float(bbox[2]) - float(bbox[0])
        bh = float(bbox[3]) - float(bbox[1])
        cx = float(bbox[0]) + bw * 0.5
        cy = float(bbox[1]) + bh * 0.5
        hw = bw * inner_ratio * 0.5
        hh = bh * inner_ratio * 0.5
        x1 = max(0, int(cx - hw))
        y1 = max(0, int(cy - hh))
        x2 = min(w_img, int(cx + hw))
        y2 = min(h_img, int(cy + hh))
    else:
        # 'full' — entire bbox
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w_img, int(bbox[2]))
        y2 = min(h_img, int(bbox[3]))

    if x2 <= x1 or y2 <= y1:
        return np.zeros(6, dtype=np.float32)

    patch = img[y1:y2, x1:x2].astype(np.float32)

    if colour_space == "lab":
        patch = _rgb_to_lab(patch)

    # Resize to fixed size (simple averaging bins → fast, no PIL needed)
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = _fast_resize(patch, patch_size)

    # Channel-wise mean and std
    means = patch.mean(axis=(0, 1))
    stds = patch.std(axis=(0, 1))
    return np.concatenate([means, stds]).astype(np.float32)


def _fast_resize(patch: np.ndarray, size: int) -> np.ndarray:
    """Simple bin-average resize (no PIL dependency)."""
    h, w = patch.shape[:2]
    h_steps = np.linspace(0, h, size, endpoint=False, dtype=np.int32)
    w_steps = np.linspace(0, w, size, endpoint=False, dtype=np.int32)
    h_bins = np.append(h_steps, h)
    w_bins = np.append(w_steps, w)
    out = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            block = patch[h_bins[i]:h_bins[i + 1], w_bins[j]:w_bins[j + 1]]
            if block.size > 0:
                out[i, j] = block.mean(axis=(0, 1))
    return out


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Fast approximate RGB → CIE Lab (D65 illuminant)."""
    # Normalise
    rgb = rgb / 255.0

    # sRGB linearise
    mask = rgb <= 0.04045
    rgb_lin = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # RGB → XYZ (D65)
    xyz = np.dot(rgb_lin, np.array([
        [0.4124564, 0.2126729, 0.0193339],
        [0.3575761, 0.7151522, 0.1191920],
        [0.1804375, 0.0721750, 0.9503041],
    ], dtype=np.float32).T)

    # XYZ → Lab
    ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz / ref
    mask = xyz > 0.008856
    f = np.where(mask, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b_val = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b_val], axis=-1)


# ── Main class ───────────────────────────────────────────────────────────
class FaceMatcher:
    """Lightweight multi-frame face matcher.

    Parameters
    ----------
    strategy : 'landmark' | 'bbox' | 'combined'
        Matching cost.
    conf_threshold : float
        Ignore detections below this confidence.
    min_frames : int | float
        Minimum frames a face must appear in.  Float < 1 ⇒ ratio of total.
    max_area_ratio : float
        Reject pairwise match when bbox areas differ more than this factor.
    max_position_shift : float
        Reject pairwise match when bbox-centre moves more than this many
        median-face-widths between consecutive frames.  Default 5.0 works
        well for ~60° orbit steps; set higher for sparse sampling.
    lm_weight : float
        Weight of landmark distance in combined cost (before quality scaling).
    pos_weight : float
        Weight of centre-position distance in combined cost.
    img_weight : float
        Weight of image-patch penalty in combined cost (0 = disabled).
        Requires image_features_list passed to match().
        Image distance is applied as a *rejection penalty*: cost is added
        only when distance exceeds img_reject_threshold, avoiding noise
        from cross-pose appearance changes (hair vs skin).
    img_reject_threshold : float
        Minimum image-feature L2 distance to trigger the penalty.
        Below this, image features contribute zero cost ("只罚不奖").
    """

    def __init__(
        self,
        strategy: str = "combined",
        conf_threshold: float = 0.5,
        min_frames: float = 2,
        max_area_ratio: float = 3.0,
        max_position_shift: float = 5.0,
        lm_weight: float = 0.3,
        pos_weight: float = 0.3,
        img_weight: float = 0.15,
        img_reject_threshold: float = 15.0,
    ):
        if strategy not in ("landmark", "bbox", "combined"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.conf_threshold = conf_threshold
        self.min_frames = min_frames
        self.max_area_ratio = max_area_ratio
        self.max_position_shift = max_position_shift
        self.lm_weight = lm_weight
        self.pos_weight = pos_weight
        self.img_weight = img_weight
        self.img_reject_threshold = img_reject_threshold

    # ── public API ───────────────────────────────────────────────────
    def match(
        self,
        bboxes_list: List[List[List[float]]],
        confs_list: List[List[float]],
        landmarks_list: List[List[np.ndarray]],
        image_features_list: Optional[List[List[np.ndarray]]] = None,
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        """Run matching.  Returns {global_id: [(frame, local_idx, conf), ...]}.

        Parameters
        ----------
        bboxes_list, confs_list, landmarks_list :
            RetinaFace outputs, one list per frame.
        image_features_list : optional
            Per-face image descriptors from extract_patch_features().
            Same nesting as bboxes_list.
        """
        n_frames = len(bboxes_list)
        use_img = image_features_list is not None and self.img_weight > 0

        # 1. filter by confidence
        kept = self._filter(confs_list)

        # 2. compute global reference scale (median face width)
        all_widths = []
        for fi in range(n_frames):
            for li in kept[fi]:
                w = float(bboxes_list[fi][li][2]) - float(bboxes_list[fi][li][0])
                if w > 0:
                    all_widths.append(w)
        ref_scale = float(np.median(all_widths)) if all_widths else 1.0

        # 3. precompute descriptors
        desc_lookup = {}   # (fi, li) → geom descriptor
        lm_qual_lookup = {}  # (fi, li) → landmark quality
        img_lookup = {}      # (fi, li) → image descriptor (if available)
        for fi in range(n_frames):
            for li in kept[fi]:
                bbox = bboxes_list[fi][li]
                ldm = landmarks_list[fi][li]
                lm = _landmark_descriptor(ldm)
                bb = _bbox_descriptor(bbox, ref_scale)
                desc_lookup[(fi, li)] = np.concatenate([lm, bb])
                lm_qual_lookup[(fi, li)] = _landmark_quality(ldm, bbox)
                if use_img:
                    img_lookup[(fi, li)] = image_features_list[fi][li]

        # 4. pairwise matching between consecutive frames
        edges = []  # (f_i, local_i, f_{i+1}, local_{i+1}, cost)
        for i in range(n_frames - 1):
            if not kept[i] or not kept[i + 1]:
                continue
            edges += self._match_pair(
                i, kept[i], kept[i + 1],
                bboxes_list[i], bboxes_list[i + 1],
                desc_lookup, lm_qual_lookup,
                img_lookup if use_img else None,
                ref_scale,
            )

        # 5. build global identities (union-find)
        identities = self._build_identities(edges, n_frames)

        # 6. enforce min_frames
        threshold = self.min_frames
        if isinstance(threshold, float) and threshold < 1:
            threshold = max(2, int(n_frames * threshold))

        result: Dict[int, List[Tuple[int, int, float]]] = {}
        for gid, apps in identities.items():
            apps.sort(key=lambda x: x[0])
            if len(apps) >= threshold:
                result[gid] = [
                    (fi, li, confs_list[fi][li]) for fi, li in apps
                ]
        return result

    # ── internals ────────────────────────────────────────────────────
    def _filter(self, confs_list):
        return [
            [j for j, c in enumerate(frame_confs) if c >= self.conf_threshold]
            for frame_confs in confs_list
        ]

    def _match_pair(self, fi, idx_a, idx_b, bboxes_a, bboxes_b,
                    desc_lookup, lm_qual_lookup, img_lookup, ref_scale):
        """Match faces between two frames. Returns edge list."""
        n_a, n_b = len(idx_a), len(idx_b)
        if n_a == 0 or n_b == 0:
            return []

        cost = np.full((max(n_a, n_b), max(n_b, n_a)), 1e6, dtype=np.float32)
        for i in range(n_a):
            for j in range(n_b):
                gi, gj = idx_a[i], idx_b[j]
                img_d = None
                if img_lookup is not None:
                    img_d = float(np.linalg.norm(
                        img_lookup[(fi, gi)] - img_lookup[(fi + 1, gj)]
                    ))
                cost[i, j] = self._cost(
                    desc_lookup[(fi, gi)],
                    desc_lookup[(fi + 1, gj)],
                    lm_qual_lookup.get((fi, gi), 1.0),
                    lm_qual_lookup.get((fi + 1, gj), 1.0),
                    img_d,
                )

        pairs = _hungarian_match(cost)
        edges = []
        for a, b in pairs:
            if a >= n_a or b >= n_b:
                continue
            gi, gj = idx_a[a], idx_b[b]

            # area-ratio gate
            ar = self._area_ratio(bboxes_a[gi], bboxes_b[gj])
            if ar > self.max_area_ratio:
                continue

            # position-shift gate
            ps = self._position_shift(bboxes_a[gi], bboxes_b[gj], ref_scale)
            if ps > self.max_position_shift:
                continue

            edges.append((fi, gi, fi + 1, gj, float(cost[a, b])))
        return edges

    def _cost(self, da, db, qual_a=1.0, qual_b=1.0, img_d=None):
        """Weighted combined cost between two face descriptors.

        Descriptor layout (8-D):
          [0:4]  landmark:  [nose_x, nose_y, mouth_x, mouth_y]
          [4:8]  bbox:      [log_area, aspect, cx, cy]

        img_d is the L2 distance between image-patch descriptors (optional).
        """
        if self.strategy == "landmark":
            return float(np.linalg.norm(da[:4] - db[:4]))

        if self.strategy == "bbox":
            return float(np.linalg.norm(da[4:] - db[4:]))

        # combined
        lm_d = float(np.linalg.norm(da[:4] - db[:4]))
        bb_area_aspect_d = float(np.linalg.norm(da[4:6] - db[4:6]))
        bb_pos_d = float(np.linalg.norm(da[6:8] - db[6:8]))

        # landmark quality gate — if either face has broken landmarks, skip
        q = min(qual_a, qual_b)
        if q < 0.5:
            q = 0.0
        effective_lm_w = self.lm_weight * q

        # image rejection penalty — only added when distance is large
        # ("只罚不奖": if two faces look similar it might be coincidence,
        #  but if they look very different they can't be the same person)
        has_img = img_d is not None and self.img_weight > 0
        img_penalty = 0.0
        if has_img and img_d > self.img_reject_threshold:
            img_penalty = self.img_weight * (img_d - self.img_reject_threshold)

        # remaining weight to bbox features (image isn't a regular feature,
        # it's a penalty on top, so no weight rebalancing needed)
        bb_w = 1.0 - effective_lm_w - self.pos_weight
        if bb_w < 0:
            bb_w = 0.0

        cost = (
            effective_lm_w * lm_d
            + bb_w * bb_area_aspect_d
            + self.pos_weight * bb_pos_d
        ) + img_penalty
        return cost

    @staticmethod
    def _area_ratio(ba, bb):
        wa = max(float(ba[2]) - float(ba[0]), 1)
        ha = max(float(ba[3]) - float(ba[1]), 1)
        wb = max(float(bb[2]) - float(bb[0]), 1)
        hb = max(float(bb[3]) - float(bb[1]), 1)
        aa, ab = wa * ha, wb * hb
        return max(aa, ab) / max(min(aa, ab), 1)

    @staticmethod
    def _position_shift(ba, bb, ref_scale):
        """Euclidean distance between bbox centres, in ref_scale units."""
        wa = max(float(ba[2]) - float(ba[0]), 1)
        ha = max(float(ba[3]) - float(ba[1]), 1)
        wb = max(float(bb[2]) - float(bb[0]), 1)
        hb = max(float(bb[3]) - float(bb[1]), 1)
        cxa = float(ba[0]) + wa * 0.5
        cya = float(ba[1]) + ha * 0.5
        cxb = float(bb[0]) + wb * 0.5
        cyb = float(bb[1]) + hb * 0.5
        s = max(ref_scale, 1.0)
        return np.sqrt((cxa - cxb) ** 2 + (cya - cyb) ** 2) / s

    @staticmethod
    def _build_identities(edges, _n_frames):
        """Union-find over (frame_idx, local_face_idx) nodes."""
        node_id = {}
        nid = 0
        for fi, li, fj, lj, _c in edges:
            for key in ((fi, li), (fj, lj)):
                if key not in node_id:
                    node_id[key] = nid
                    nid += 1
        parent = list(range(nid))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for fi, li, fj, lj, _c in edges:
            union(node_id[(fi, li)], node_id[(fj, lj)])

        groups = defaultdict(list)
        for (fi, li), n in node_id.items():
            groups[find(n)].append((fi, li))
        return dict(groups)

    # ── reporting ────────────────────────────────────────────────────
    def report(self, matches: Dict, bboxes_list=None) -> str:
        """Human-readable summary string."""
        if not matches:
            return "No consistent faces found across frames."
        n_frames = max(a[0] for apps in matches.values() for a in apps) + 1
        lines = [
            f"Input: {n_frames} frames  →  matched {len(matches)} consistent "
            f"face(s) (strategy={self.strategy}, conf≥{self.conf_threshold}):\n"
        ]
        for gid, apps in sorted(matches.items()):
            frames = [a[0] for a in apps]
            avg_c = np.mean([a[2] for a in apps])
            line = (
                f"  Face #{gid}:  spans {len(apps)}/{n_frames} frames  "
                f"avg-conf={avg_c:.3f}  |  frames={frames}"
            )
            if bboxes_list is not None:
                bbox_strs = []
                for fi, li, _ in apps:
                    b = bboxes_list[fi][li]
                    bbox_strs.append(
                        f"f{fi}:[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"
                    )
                line += f"\n           bboxes: {'  '.join(bbox_strs)}"
            lines.append(line)
        return "\n".join(lines)


# ── Convenience one-liner ────────────────────────────────────────────────
def match_faces(
    bboxes_list: List[List[List[float]]],
    confs_list: List[List[float]],
    landmarks_list: List[List[np.ndarray]],
    conf_threshold: float = 0.5,
    min_frames: float = 0.8,
    strategy: str = "combined",
    **kwargs,
) -> Dict[int, List[Tuple[int, int, float]]]:
    """One-call convenience wrapper."""
    m = FaceMatcher(
        strategy=strategy,
        conf_threshold=conf_threshold,
        min_frames=min_frames,
        **kwargs,
    )
    return m.match(bboxes_list, confs_list, landmarks_list)
