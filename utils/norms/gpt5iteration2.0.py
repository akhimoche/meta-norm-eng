# utils/norms/improved_norm.py
# Rotating Quadrant Rest + Sanctuaries (RQR-S) Norm
#
# Absolute imports as required by the integration code.
from typing import Set, Iterable
from utils.norms.norm import Norm, Coord


class RotatingQuadrantSanctuaryNorm(Norm):
    """
    Rotating Quadrant Rest + Sanctuaries (RQR-S)

    Purpose
    -------
    Maximize long-run social welfare in Commons Harvest Open by:
      1) Keeping a few apples in every patch permanently protected (sanctuaries) to
         guarantee 3+ neighbors somewhere in each cluster → highest regrowth band.
      2) Applying a *rotating quadrant closure* that alternately blocks entire
         patch groups (their apple tiles only), giving them structured recovery
         windows while two opposite quadrants always remain open.

    Why this beats gate-throttling
    ------------------------------
    The previous norm used fences and narrow gates. That added travel/queuing costs,
    delayed early harvesting, and did not give specific patches a true rest. Here we
    eliminate travel bottlenecks and instead rest apple tiles directly, which
    (a) shortens paths, (b) starts collection immediately, and (c) yields sustained,
    denser respawn thanks to predictable off-cycles.

    Behaviour
    ---------
    - Sanctuaries: a small set of apples per patch are *always* blocked.
    - Rotation: time is split into equal windows. In each window one quadrant pair
      is open and the orthogonal pair is closed. Closed = apple tiles in those
      quadrants are blocked (agents won’t step on/eat them). We alternate the two
      pairs so each quadrant is open 50% of the time and resting 50% of the time.
    - Self-contained: requires only timestep `t`. No external state.

    Hyperparameters
    ---------------
    window_len : int
        Length of one window (timesteps) before switching open/closed pairs.
        Defaults to 60 (balanced rest vs access).
    cycle_pairs : tuple[tuple[str, str], tuple[str, str]]
        Two diagonal pairs that alternate: (("UL", "LR"), ("UR", "LL")).
    sanctuaries : Set[Coord]
        Permanently protected apple tiles.
    quadrant_patches : dict[str, Set[Coord]]
        Apple tiles per quadrant (UL/UR/LL/LR). Used for rotating closures.

    Usage
    -----
    Instantiate with `RotatingQuadrantSanctuaryNorm(epsilon=0.0)` and pass to agents.
    Agents combine the returned blocked set with physical obstacles for A*.
    """

    # --- Tuned defaults for the provided 24x18 Commons Harvest Open map ---
    MAP_W: int = 24
    MAP_H: int = 18

    # Rotation window length (steps). 60 gives each closed quadrant a meaningful rest
    # while keeping access frequent enough to avoid idling.
    window_len: int = 60

    # The two alternating diagonal pairs that remain OPEN each window.
    # Window 0: UL + LR open; UR + LL closed.
    # Window 1: UR + LL open; UL + LR closed; repeat.
    cycle_pairs = (("UL", "LR"), ("UR", "LL"))

    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__("rotating_quadrant_sanctuary", epsilon)

        # ---- All apple coordinates grouped by patch (from prompt) ----
        UL_corner = {(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3)}
        UL_mid = {
            (8, 1), (7, 2), (8, 2), (9, 2), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
            (7, 4), (8, 4), (9, 4), (8, 5)
        }
        LL_patch = {
            (3, 6), (2, 7), (3, 7), (4, 7),
            (1, 8), (2, 8), (3, 8), (4, 8), (5, 8),
            (2, 9), (3, 9), (4, 9),
            (3, 10)
        }
        UR_mid = {
            (15, 1), (14, 2), (15, 2), (16, 2), (13, 3), (14, 3), (15, 3), (16, 3),
            (17, 3), (14, 4), (15, 4), (16, 4), (15, 5)
        }
        LR_patch = {
            (20, 6), (19, 7), (20, 7), (21, 7),
            (18, 8), (19, 8), (20, 8), (21, 8), (22, 8),
            (19, 9), (20, 9), (21, 9),
            (20, 10)
        }
        UR_corner = {(20, 1), (21, 1), (22, 1), (21, 2), (22, 2), (22, 3)}

        # Group patches into quadrants (apple tiles only)
        self.quadrant_patches = {
            "UL": set().union(UL_corner, UL_mid),
            "UR": set().union(UR_corner, UR_mid),
            "LL": set(LL_patch),
            "LR": set(LR_patch),
        }

        # ---- Permanent sanctuaries (3 per cluster; 4 for big clusters) ----
        # Picked to be central to each patch to maximize neighbor coverage.
        self.sanctuaries: Set[Coord] = {
            # UL corner + UL mid
            (2, 1), (1, 2), (2, 2),        # UL corner
            (7, 3), (8, 3), (9, 3), (8, 4),  # UL mid (4 seeds → robust)
            # LL
            (2, 8), (3, 8), (4, 8),
            # UR mid + UR corner
            (14, 3), (15, 3), (16, 3), (15, 4),  # UR mid
            (21, 1), (21, 2), (22, 2),           # UR corner
            # LR
            (19, 8), (20, 8), (21, 8),
        }

        # Sanity: make sure sanctuaries are actual apples
        all_apples = set().union(*self.quadrant_patches.values())
        assert self.sanctuaries <= all_apples

    # ----------------------------- Public API ---------------------------------
    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return the norm-blocked (col, row) coordinates at timestep t.

        - Always block sanctuary tiles (never harvested).
        - Determine which diagonal pair is OPEN this window; all apple tiles in the
          *other* pair are blocked to rest. Switch every `window_len` steps.
        """
        blocked: Set[Coord] = set(self.sanctuaries)

        # Determine which pair is currently OPEN
        window = (t // self.window_len) % len(self.cycle_pairs)
        open_quads = self.cycle_pairs[window]  # e.g., ("UL", "LR")

        # The quads to CLOSE are the complement
        to_close = {"UL", "UR", "LL", "LR"} - set(open_quads)

        # Block all apple tiles in the closed quadrants
        for q in to_close:
            blocked |= self.quadrant_patches[q]

        return blocked

    # -------------------------- Convenience alias -----------------------------
    # Some runner scripts expect a class named `NormClass`
    # (kept for drop-in compatibility with your experiment harness).
    pass


# Expose a convenient alias for experimenters
NormClass = RotatingQuadrantSanctuaryNorm

# Human-readable meta information (optional)
meta_norm = {
    "verbal_explanation": (
        "Rotating Quadrant Rest + Sanctuaries: permanently protect a few apples in every "
        "patch to keep local density high, and then alternate which diagonal pairs of "
        "patches are open. The closed quadrants have their apple tiles blocked so they "
        "can regrow; the open quadrants are fully accessible. No fences, no gates—just "
        "shorter paths and targeted rest where it matters."
    ),
    "reasoning": (
        "The prior norm achieved stability but at a lower slope due to pulsed fence "
        "bottlenecks and a long initial delay. By removing travel throttles and applying "
        "structured, rotating rest directly to apple tiles, agents harvest sooner and "
        "spend more time on apples that have benefited from recovery."
    ),
    "code_with_placeholders": r'''
# Instantiate with:
#   from utils.norms.improved_norm import NormClass
#   norm = NormClass(epsilon=0.0)
''',
    "hyperparameters_for_this_environment": {
        "name": "rotating_quadrant_sanctuary",
        "map_size": [24, 18],
        "window_len": 60,
        "cycle_pairs": (("UL", "LR"), ("UR", "LL")),
        "epsilon_default": 0.0,
        "sanctuaries_count": 3,
    },
}
