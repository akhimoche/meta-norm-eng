# utils/norms/improved_norm.py
# Tri-Phase Lanes + Rotating Fallow Norm for Commons Harvest Open

from typing import Set
from utils.norms.norm import Norm, Coord

class TriPhaseRotatingFallowNorm(Norm):
    """
    Tri-Phase Rotating Fallow Norm

    Overview
    --------
    This norm combines (A) lighter, more frequent, three-gate horizontal fences that
    stagger inflows without starving the map, with (B) patch-internal rotating fallow
    schedules (checkerboard parity by (x+y)%2) to create repeated rest periods that
    amplify density-dependent regrowth. Each patch also retains a *tiny* permanent core
    (2 cells) so clusters never fully collapse.

    Why it helps
    ------------
    - Regrowth radius is 2 and jumps to 2.5%/step with 3+ nearby apples.
      Rotating half the patch off at a time yields regular "reseed + rebound" waves,
      while the 2-cell permanent core ensures the patch cannot crash to zero.
    - Tri-phase (left, center, right) gates with shorter cycles reduce bunching,
      cut travel congestion, and get fruit flowing earlier (top fence opens at t=60).
    - Per-patch phase offsets desynchronise rest windows, smoothing global harvests.

    Plug-and-play
    -------------
    Instantiate with `TriPhaseRotatingFallowNorm(epsilon=0.0)`. The norm is purely
    time-based: agents call `get_blocked_positions(t)` and treat returned coordinates
    as additional obstacles. No global state is read or written.

    Map & Symbols
    -------------
    - Coordinates: (col, row), 0-indexed, 24×18 map.
    - Patches are built directly from the prompt’s coordinates.
    """

    # Map dimensions
    MAP_W: int = 24
    MAP_H: int = 18

    # ---------------- Fence hyperparameters ----------------
    # Rows for fences (same placement as prior norm)
    main_fence_row: int = 12
    top_fence_row: int = 6

    # Tri-phase cycle: LEFT -> CENTER -> RIGHT -> (repeat), with rests between opens.
    # Shorter, more frequent pulses to smooth flow; wider gates to cut congestion.
    main_cycle: int = 60
    main_open_window: int = 12
    main_rest_window: int = 8
    main_start_delay: int = 8

    top_start_delay: int = 60
    top_cycle: int = 60
    top_open_window: int = 10
    top_rest_window: int = 10

    # Gate columns (wider; add a center gate to distribute agents)
    main_left_gate_cols  = [4, 5, 6, 7]
    main_center_gate_cols = [10, 11, 12, 13]
    main_right_gate_cols = [16, 17, 18, 19]

    top_left_gate_cols   = [6, 7, 8]
    top_center_gate_cols = [11, 12]
    top_right_gate_cols  = [15, 16, 17]

    # ---------------- Rotating fallow hyperparameters ----------------
    # Flip which parity is blocked every `phase_len` steps.
    # Stagger patches with fixed offsets so they don't all rest at once.
    phase_len_big: int = 40      # for the two big mid patches
    phase_len_mid: int = 36      # for the two lower patches
    phase_len_small: int = 28    # for the two corner patches

    # Permanent "seed" cores (two tiles per patch). These are *always* blocked.
    # They sit near the densest part of each cluster to anchor regrowth.
    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__("tri_phase_rotating_fallow", epsilon)

        # ---- Build patch coordinate sets from the prompt ----
        # Upper-left corner patch (6 tiles)
        ul_corner = {
            (1, 1), (2, 1), (3, 1),
            (1, 2), (2, 2),
            (1, 3),
        }

        # Upper center-left (diamond, 13 tiles)
        ucl = {
            (8, 1),
            (7, 2), (8, 2), (9, 2),
            (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
            (7, 4), (8, 4), (9, 4),
            (8, 5),
        }

        # Upper center-right (diamond, 13 tiles)
        ucr = {
            (15, 1),
            (14, 2), (15, 2), (16, 2),
            (13, 3), (14, 3), (15, 3), (16, 3), (17, 3),
            (14, 4), (15, 4), (16, 4),
            (15, 5),
        }

        # Lower-left patch (diamond-like, 13 tiles)
        ll = {
            (3, 6),
            (2, 7), (3, 7), (4, 7), (5, 7),
            (2, 8), (3, 8), (4, 8),
            (2, 9), (3, 9), (4, 9),
            (3, 10),
        }

        # Lower-right patch (diamond-like, 13 tiles)
        lr = {
            (20, 6),
            (19, 7), (20, 7), (21, 7),
            (18, 8), (19, 8), (20, 8), (21, 8), (22, 8),
            (19, 9), (20, 9), (21, 9),
            (20, 10),
        }

        # Upper-right corner patch (6 tiles)
        ur_corner = {
            (20, 1), (21, 1), (22, 1),
            (21, 2), (22, 2),
            (22, 3),
        }

        # Record patches and sizes
        self._patches = [
            ("ul_small", ul_corner, "small", 0),
            ("ucl_big", ucl, "big", 1),
            ("ucr_big", ucr, "big", 2),
            ("ll_mid", ll, "mid", 3),
            ("lr_mid", lr, "mid", 4),
            ("ur_small", ur_corner, "small", 5),
        ]

        # Choose tiny 2-tile permanent cores near centers (always blocked)
        # (Small patches: pick two adjacent; bigger patches: center + one neighbor)
        self._permanent_cores: Set[Coord] = {
            # UL corner (anchor the 2-center)
            (2, 1), (2, 2),
            # Upper center-left big (center column 8)
            (8, 3), (8, 2),
            # Upper center-right big (center column 15)
            (15, 3), (15, 2),
            # Lower-left mid (center column 3)
            (3, 8), (3, 7),
            # Lower-right mid (center column 20)
            (20, 8), (20, 7),
            # UR corner
            (21, 1), (22, 2),
        }

        # Precompute parity sets for each patch to support rotating fallow
        self._patch_parities = []
        for name, coords, ptype, idx in self._patches:
            even = {c for c in coords if ((c[0] + c[1]) % 2 == 0)} - self._permanent_cores
            odd  = (coords - even) - self._permanent_cores
            phase_len = (
                self.phase_len_big if ptype == "big"
                else self.phase_len_mid if ptype == "mid"
                else self.phase_len_small
            )
            # Stagger: each patch gets a fixed offset in cycles to desynchronise
            offset = idx  # simple, effective: 0..5
            self._patch_parities.append({
                "name": name,
                "even": even,
                "odd": odd,
                "phase_len": phase_len,
                "offset": offset,
            })

        # ---------------- Precompute fences ----------------
        self._main_fence_full = {(c, self.main_fence_row) for c in range(1, self.MAP_W - 1)}
        self._top_fence_full  = {(c, self.top_fence_row)  for c in range(1, self.MAP_W - 1)}

        # Gate sets per fence (three gates each)
        self._main_left_gate   = {(c, self.main_fence_row) for c in self.main_left_gate_cols}
        self._main_center_gate = {(c, self.main_fence_row) for c in self.main_center_gate_cols}
        self._main_right_gate  = {(c, self.main_fence_row) for c in self.main_right_gate_cols}

        self._top_left_gate    = {(c, self.top_fence_row) for c in self.top_left_gate_cols}
        self._top_center_gate  = {(c, self.top_fence_row) for c in self.top_center_gate_cols}
        self._top_right_gate   = {(c, self.top_fence_row) for c in self.top_right_gate_cols}

        # Defensive checks (avoid outer walls)
        assert all(1 <= c <= self.MAP_W - 2 for c, _ in self._main_fence_full)
        assert all(1 <= c <= self.MAP_W - 2 for c, _ in self._top_fence_full)

    # ---------------- Public API ----------------
    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return the set of coordinates blocked by the norm at timestep t.
        """
        blocked: Set[Coord] = set()

        # (A) Permanent tiny cores (always blocked)
        blocked |= self._permanent_cores

        # (B) Rotating fallow schedules per patch (checkerboard parity flips)
        for p in self._patch_parities:
            phase_len = p["phase_len"]
            if phase_len <= 0:
                continue
            # Which parity is OFF right now? Flip every 'phase_len' steps with an offset.
            flip = ((t // phase_len) + p["offset"]) % 2
            if flip == 0:
                blocked |= p["even"]
            else:
                blocked |= p["odd"]

        # (C) Fences with tri-phase pulsing
        blocked |= self._main_fence_positions(t)
        blocked |= self._top_fence_positions(t)

        return blocked

    # ---------------- Internal helpers ----------------
    def _tri_phase(self, t: int, start_delay: int, cycle: int,
                   open_window: int, rest_window: int,
                   fence_full: Set[Coord],
                   left_gate: Set[Coord], center_gate: Set[Coord], right_gate: Set[Coord]) -> Set[Coord]:
        """
        Generic tri-phase pulsing: LEFT open -> rest -> CENTER open -> rest -> RIGHT open -> rest.
        """
        if t < start_delay:
            return set(fence_full)

        phase = (t - start_delay) % cycle

        # phase segments (6 segments total)
        seg1_end = open_window                        # LEFT open
        seg2_end = seg1_end + rest_window             # rest
        seg3_end = seg2_end + open_window             # CENTER open
        seg4_end = seg3_end + rest_window             # rest
        seg5_end = seg4_end + open_window             # RIGHT open
        seg6_end = seg5_end + rest_window             # rest (== cycle end)

        blocked = set(fence_full)
        if phase < seg1_end:
            blocked -= left_gate
        elif phase < seg2_end:
            pass
        elif phase < seg3_end:
            blocked -= center_gate
        elif phase < seg4_end:
            pass
        elif phase < seg5_end:
            blocked -= right_gate
        elif phase < seg6_end:
            pass
        else:
            pass

        return blocked

    def _main_fence_positions(self, t: int) -> Set[Coord]:
        return self._tri_phase(
            t,
            self.main_start_delay, self.main_cycle,
            self.main_open_window, self.main_rest_window,
            self._main_fence_full,
            self._main_left_gate, self._main_center_gate, self._main_right_gate
        )

    def _top_fence_positions(self, t: int) -> Set[Coord]:
        return self._tri_phase(
            t,
            self.top_start_delay, self.top_cycle,
            self.top_open_window, self.top_rest_window,
            self._top_fence_full,
            self._top_left_gate, self._top_center_gate, self._top_right_gate
        )


# Expose a convenient alias for experiment runners
NormClass = TriPhaseRotatingFallowNorm

# Optional meta for reference (non-functional)
meta_norm = {
    "verbal_explanation": (
        "Tri-Phase Rotating Fallow: three pulsing gates (left/center/right) at rows 6 and 12 "
        "stagger inflows with short, frequent openings. Inside each patch, half the tiles are "
        "blocked at a time by parity, flipping every few dozen steps with per-patch phase offsets, "
        "so apples rest and regrow in waves while a tiny 2-tile core remains permanently protected."
    ),
    "hyperparameters": {
        "main_row": 12, "top_row": 6,
        "main_cycle": 60, "main_open": 12, "main_rest": 8, "main_start": 8,
        "top_cycle": 60, "top_open": 10, "top_rest": 10, "top_start": 60,
        "main_left_cols":  [4,5,6,7],
        "main_center_cols":[10,11,12,13],
        "main_right_cols": [16,17,18,19],
        "top_left_cols":   [6,7,8],
        "top_center_cols": [11,12],
        "top_right_cols":  [15,16,17],
        "phase_len_big": 40, "phase_len_mid": 36, "phase_len_small": 28,
        "permanent_core_count_per_patch": 2
    }
}
