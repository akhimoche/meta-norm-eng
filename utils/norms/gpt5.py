# utils/norms/gpt5.py
# Pulse-Gated Sanctuary Norm for Commons Harvest Open environment

from typing import Set
from utils.norms.norm import Norm, Coord

class PulseGatedSanctuaryNorm(Norm):
    """
    Pulse-Gated Sanctuary Norm

    Overview
    --------
    A self-contained, plug-and-play norm that:
      1) Permanently blocks a small, hand-picked set of 'sanctuary' apple tiles in each patch
         so at least ~3 apples always remain and keep local regrowth in the highest band.
      2) Adds a MAIN horizontal fence at row 12 with two narrow gates (left/right) that open
         in pulses to throttle the number of agents entering the orchards from the P spawns.
      3) Adds a TOP horizontal fence at row 6 with two narrow gates near the Q spawns; this
         fence is closed for an initial delay and then pulses, slowing the two inside agents.

    Why this works
    --------------
    - Density-dependent regrowth collapses if a cluster hits zero. Keeping a few apples
      permanently uneaten preserves a 3+ neighbor regime (2.5% respawn/step), stabilizing patches.
    - Pulsed gates prevent the whole population from rushing the same cluster simultaneously,
      distributing harvests across time and space.

    Hyperparameters (defaults tuned for Commons Harvest Open map)
    -------------------------------------------------------------
    - main_fence_row: int = 12
    - top_fence_row: int = 6
    - main_cycle: int = 80                 # full cycle length for main fence
    - main_open_window: int = 20           # steps the left/right gate is open
    - main_rest_window: int = 20           # recovery period between gate openings
    - main_start_delay: int = 10           # initial settling before first opening
    - top_start_delay: int = 150           # keep top fence closed early to spare top orchards
    - top_cycle: int = 80
    - top_open_window: int = 10
    - top_rest_window: int = 30
    - gate_columns: dict with lists        # narrow columns for each gate opening
      • main_left_gate_cols  = [5, 6, 7]
      • main_right_gate_cols = [16, 17, 18]
      • top_left_gate_cols   = [6, 7]
      • top_right_gate_cols  = [16, 17]

    Usage
    -----
    Instantiate with `PulseGatedSanctuaryNorm(epsilon=0.0)` and pass into your agents.
    Agents will call `get_blocked_positions(t)` each step and treat the returned set
    as additional obstacles alongside walls.

    Notes
    -----
    - Coordinates are (col, row), zero-indexed, matching the provided ASCII map.
    - This norm does not require or consume any environment state beyond timestep t.
    - Epsilon is respected by the base Norm (0.0 in your tests means always comply).
    """

    # Map dimensions (from prompt)
    MAP_W: int = 24
    MAP_H: int = 18

    # --- FENCE & SCHEDULE HYPERPARAMETERS (safe defaults for this environment) ---
    main_fence_row: int = 12
    top_fence_row: int = 6

    main_cycle: int = 80
    main_open_window: int = 20
    main_rest_window: int = 20
    main_start_delay: int = 10

    top_start_delay: int = 150
    top_cycle: int = 80
    top_open_window: int = 10
    top_rest_window: int = 30

    # Gate columns (narrow)
    main_left_gate_cols = [5, 6, 7]
    main_right_gate_cols = [16, 17, 18]
    top_left_gate_cols = [6, 7]
    top_right_gate_cols = [16, 17]

    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__("pulse_gated_sanctuary", epsilon)

        # --- PERMANENT SANCTUARIES ---
        # Hand-picked 'seed' apples to preserve per patch (never enter/harvest these tiles).
        # These are all valid apple coordinates from the prompt; we protect ~3 per cluster.
        self._sanctuaries: Set[Coord] = {
            # Upper-left corner patch
            (2, 1), (1, 2), (2, 2),
            # Upper-left (mid-left) big patch
            (7, 3), (8, 3), (9, 3),
            # Lower-left patch
            (2, 8), (3, 8), (4, 8),
            # Upper-right (mid-right) big patch
            (14, 3), (15, 3), (16, 3),
            # Lower-right patch
            (19, 8), (20, 8), (21, 8),
            # Upper-right corner patch
            (21, 1), (21, 2), (22, 2),
        }

        # Precompute full horizontal fence coordinate sets (excluding outer walls)
        self._main_fence_full = {(c, self.main_fence_row) for c in range(1, self.MAP_W - 1)}
        self._top_fence_full = {(c, self.top_fence_row) for c in range(1, self.MAP_W - 1)}

        # Convert gate column lists to sets at the appropriate fence rows
        self._main_left_gate = {(c, self.main_fence_row) for c in self.main_left_gate_cols}
        self._main_right_gate = {(c, self.main_fence_row) for c in self.main_right_gate_cols}
        self._top_left_gate = {(c, self.top_fence_row) for c in self.top_left_gate_cols}
        self._top_right_gate = {(c, self.top_fence_row) for c in self.top_right_gate_cols}

        # Defensive: ensure we never accidentally block the outer walls (0 or MAP_W-1)
        assert all(1 <= c <= self.MAP_W - 2 for c, _ in self._main_fence_full)
        assert all(1 <= c <= self.MAP_W - 2 for c, _ in self._top_fence_full)

    # --- Public API expected by the agents -------------------------------------------------
    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return a set of (col, row) coordinates that are blocked by the norm at timestep t.
        This is combined by the agent with physical obstacles for A* pathfinding.
        """
        blocked: Set[Coord] = set()

        # 1) Permanent sanctuaries (always blocked)
        blocked |= self._sanctuaries

        # 2) Main fence at row 12: pulsed left/right gates
        blocked |= self._main_fence_positions(t)

        # 3) Top fence at row 6: delayed + pulsed gates near the Q spawns
        blocked |= self._top_fence_positions(t)

        return blocked

    # --- Internal helpers -----------------------------------------------------------------
    def _main_fence_positions(self, t: int) -> Set[Coord]:
        """
        Compute blocked coords for the main fence at row 12 with pulsed gates.
        Before main_start_delay, the fence is fully closed (no gates open).
        Thereafter, the cycle is:
            [open left for main_open_window]
            [closed for main_rest_window]
            [open right for main_open_window]
            [closed for main_rest_window]
        """
        if t < self.main_start_delay:
            # fully closed
            return set(self._main_fence_full)

        # Determine phase within the main cycle
        phase = (t - self.main_start_delay) % self.main_cycle
        left_open_span = self.main_open_window
        left_rest_end = left_open_span + self.main_rest_window
        right_open_end = left_rest_end + self.main_open_window
        full_cycle_end = right_open_end + self.main_rest_window  # should equal main_cycle

        # Start with full fence, then remove whichever gate is open in this phase
        blocked = set(self._main_fence_full)
        if phase < left_open_span:
            # Open LEFT gate
            blocked -= self._main_left_gate
        elif phase < left_rest_end:
            # Both closed (recovery)
            pass
        elif phase < right_open_end:
            # Open RIGHT gate
            blocked -= self._main_right_gate
        elif phase < full_cycle_end:
            # Both closed (recovery)
            pass
        else:
            # Shouldn't happen, but keep closed
            pass
        return blocked

    def _top_fence_positions(self, t: int) -> Set[Coord]:
        """
        Compute blocked coords for the top fence at row 6.
        For t < top_start_delay: fully closed.
        After that, pulse short openings near the Q spawns:
            [open left gate for top_open_window]
            [closed for top_rest_window]
            [open right gate for top_open_window]
            [closed for top_rest_window]
        """
        if t < self.top_start_delay:
            return set(self._top_fence_full)

        phase = (t - self.top_start_delay) % self.top_cycle
        left_open_span = self.top_open_window
        left_rest_end = left_open_span + self.top_rest_window
        right_open_end = left_rest_end + self.top_open_window
        full_cycle_end = right_open_end + self.top_rest_window  # equals top_cycle

        blocked = set(self._top_fence_full)
        if phase < left_open_span:
            blocked -= self._top_left_gate
        elif phase < left_rest_end:
            pass
        elif phase < right_open_end:
            blocked -= self._top_right_gate
        elif phase < full_cycle_end:
            pass
        else:
            pass
        return blocked

# Meta information for reference
meta_norm = {
    "verbal_explanation": (
        "Pulse-Gated Sanctuary Norm: we create two timed 'fences' that throttle entry to "
        "the orchards and permanently protect a few 'seed' apples inside each patch. "
        "A main fence across row 12 opens small left/right gates in pulses so not all "
        "agents flood the top at once. A secondary fence across row 6 delays access from "
        "the inside spawns to the fragile topmost orchards. Sanctuaries (blocked apple "
        "tiles) keep a handful of apples forever uneaten so neighboring tiles retain a "
        "high regrowth probability (≥3 neighbors → 2.5% per step). Together this prevents "
        "early overharvest while still letting agents efficiently collect apples over time."
    ),
    "reasoning": (
        "In this map, five selfish A* agents quickly wipe out apples (~60 steps) because "
        "everyone beelines to the same dense clusters. Regrowth needs nearby apples; after "
        "a cluster collapses, density-dependent respawn falls to near zero. Without live "
        "state, the best lever is movement constraints via blocked coordinates. We therefore: "
        "(1) Permanently preserve 3 'seed' apples in each patch so densities never hit zero, "
        "maintaining 2.5% regrowth in their neighborhood. "
        "(2) Insert a horizontal fence at row 12 (above the P spawns) with narrow gates that "
        "alternate (left/right) in short pulses separated by recovery rests. This staggers "
        "traffic so collection is spread out both spatially and temporally. "
        "(3) Add a second fence at row 6 (just below the top orchards) that remains closed "
        "for an initial delay to keep the two Q-spawn agents from instantly emptying the "
        "most fragile top clusters; after the delay it also pulses small gates near those "
        "spawns. All parameters are static and time-based, requiring no external state. "
        "Epsilon=0.0 ensures full compliance, maximizing long-run social welfare by trading "
        "a small early slowdown for sustained, regenerating harvests."
    ),
    "code_with_placeholders": r'''


# Expose a convenient alias for experimenters
NormClass = PulseGatedSanctuaryNorm
''',
    "hyperparameters_for_this_environment": {
        "name": "pulse_gated_sanctuary",
        "map_size": [24, 18],
        "main_fence_row": 12,
        "top_fence_row": 6,
        "main_cycle": 80,
        "main_open_window": 20,
        "main_rest_window": 20,
        "main_start_delay": 10,
        "top_start_delay": 150,
        "top_cycle": 80,
        "top_open_window": 10,
        "top_rest_window": 30,
        "main_left_gate_cols": [5, 6, 7],
        "main_right_gate_cols": [16, 17, 18],
        "top_left_gate_cols": [6, 7],
        "top_right_gate_cols": [16, 17],
        "sanctuaries": sorted([
            (2, 1), (1, 2), (2, 2),
            (7, 3), (8, 3), (9, 3),
            (2, 8), (3, 8), (4, 8),
            (14, 3), (15, 3), (16, 3),
            (19, 8), (20, 8), (21, 8),
            (21, 1), (21, 2), (22, 2),
        ]),
        "epsilon_default": 0.0
    }
}

