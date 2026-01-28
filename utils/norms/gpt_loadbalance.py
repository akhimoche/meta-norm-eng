"""
Load-Balancing Detours Norm for Commons Harvest.

A continuous route-shaping norm that places small, time-rotating 'speed-bump' 
obstacle clusters to bias A* paths toward different regions over time.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from utils.norms.norm import Norm, Coord


ASCII_MAP = """
WWWWWWWWWWWWWWWWWWWWWWWW
WAAA    A      A    AAAW
WAA    AAA    AAA    AAW
WA    AAAAA  AAAAA    AW
W      AAA    AAA      W
W       A      A       W
W  A                A  W
W AAA  Q        Q  AAA W
WAAAAA            AAAAAW
W AAA              AAA W
W  A                A  W
W                      W
W                      W
W                      W
W  PPPPPPPPPPPPPPPPPP  W
W PPPPPPPPPPPPPPPPPPPP W
WPPPPPPPPPPPPPPPPPPPPPPW
WWWWWWWWWWWWWWWWWWWWWWWW
"""


def _parse_ascii_map(ascii_map: str) -> Dict[str, Set[Coord]]:
    lines = [ln for ln in ascii_map.strip("\n").splitlines()]
    h = len(lines)
    w = len(lines[0]) if h > 0 else 0

    walls: Set[Coord] = set()
    apples: Set[Coord] = set()
    spaces: Set[Coord] = set()
    q_spawns: Set[Coord] = set()
    p_spawns: Set[Coord] = set()

    for y, row in enumerate(lines):
        if len(row) != w:
            raise ValueError(f"Non-rectangular ASCII map at row {y}: expected width {w}, got {len(row)}")
        for x, ch in enumerate(row):
            c: Coord = (x, y)
            if ch == "W":
                walls.add(c)
            elif ch == "A":
                apples.add(c)
            elif ch == " ":
                spaces.add(c)
            elif ch == "Q":
                q_spawns.add(c)
                spaces.add(c)
            elif ch == "P":
                p_spawns.add(c)
                spaces.add(c)
            else:
                spaces.add(c)

    return {
        "walls": walls,
        "apples": apples,
        "spaces": spaces,
        "q_spawns": q_spawns,
        "p_spawns": p_spawns,
        "width": {(w, 0)},
        "height": {(h, 0)},
    }


def _wh_from_parsed(parsed: Dict[str, Set[Coord]]) -> Tuple[int, int]:
    w = next(iter(parsed["width"]))[0]
    h = next(iter(parsed["height"]))[0]
    return w, h


def _rect(x0: int, x1: int, y0: int, y1: int) -> Set[Coord]:
    # Inclusive bounds.
    out: Set[Coord] = set()
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            out.add((x, y))
    return out


class LoadBalancingDetoursNorm(Norm):
    """
    LoadBalancingDetoursNorm (Commons Harvest Open)

    Behaviour
    ----------
    A stateless, phase-rotating detour system that biases A* routes through one of three
    broad "lanes" (left / center / right) in the empty corridor rows (11–13).

    It never hard-closes the world. Instead, it places small "speed-bump" obstacle clusters
    (blocked space tiles) that make two lanes slightly less convenient, encouraging different
    agents to distribute across map regions and patches over time.

    Phase schedule (repeating)
    --------------------------
    0) LEFT  : block center+right speed-bumps (left lane easiest)
    1) RIGHT : block left+center speed-bumps (right lane easiest)
    2) CENTER: block left+right speed-bumps (center lane easiest)
    3) MIX   : minimal bumps (all lanes usable; gentle anti-zigzag bumps)

    Hyperparameters
    ---------------
    phase_len : int
        Steps per phase before rotating to the next lane bias.
    lane_rows : Tuple[int, int]
        Corridor row range (inclusive) where bumps are placed.
    left_lane_cols / center_lane_cols / right_lane_cols : Tuple[int, ...]
        Columns defining each lane's bump footprint (bumps are placed on these columns).
    bump_thickness : int
        How many consecutive rows inside lane_rows to use for bump placement (from the top of lane_rows).
    mix_extra_bumps : bool
        If True, add small alternating bumps in MIX to reduce synchronized zig-zagging.

    Notes
    -----
    - Blocks only *space/spawn tiles*, never apple tiles.
    - Avoids blocking Q and P spawn tiles to prevent start-state failures.
    - Fully self-contained: no external state required.
    """

    def __init__(
        self,
        epsilon: float = 0.0,
        *,
        phase_len: int = 25,  # Faster rotation = less time to exploit any lane
        lane_rows: Tuple[int, int] = (4, 13),  # Extended up to row 4 (into apple area)
        left_lane_cols: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9),  # Very wide lanes
        center_lane_cols: Tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 15),  # Overlapping = more blocking
        right_lane_cols: Tuple[int, ...] = (14, 15, 16, 17, 18, 19, 20, 21),  # Very wide lanes
        bump_thickness: int = 10,  # Block full 10 rows
        mix_extra_bumps: bool = True,
    ):
        super().__init__("load_balancing_detours", epsilon)

        parsed = _parse_ascii_map(ASCII_MAP)
        w, h = _wh_from_parsed(parsed)

        self._spaces = parsed["spaces"]
        self._apples = parsed["apples"]
        self._q_spawns = parsed["q_spawns"]
        self._p_spawns = parsed["p_spawns"]
        self._w = w
        self._h = h

        self.phase_len = int(phase_len)
        self.lane_rows = (int(lane_rows[0]), int(lane_rows[1]))
        self.left_lane_cols = tuple(int(c) for c in left_lane_cols)
        self.center_lane_cols = tuple(int(c) for c in center_lane_cols)
        self.right_lane_cols = tuple(int(c) for c in right_lane_cols)
        self.bump_thickness = max(1, int(bump_thickness))
        self.mix_extra_bumps = bool(mix_extra_bumps)

        # Blockable tiles are walkable non-apple spaces, excluding spawn tiles.
        self._blockable: Set[Coord] = set(self._spaces) - set(self._apples) - set(self._q_spawns) - set(self._p_spawns)

        # Precompute bump sets for each lane.
        y0, y1 = self.lane_rows
        y0 = max(1, min(y0, h - 2))
        y1 = max(1, min(y1, h - 2))
        if y1 < y0:
            y0, y1 = y1, y0

        bump_rows = list(range(y0, min(y1, y0 + self.bump_thickness - 1) + 1))

        def lane_bumps(cols: Tuple[int, ...]) -> Set[Coord]:
            s: Set[Coord] = set()
            for y in bump_rows:
                for x in cols:
                    c: Coord = (x, y)
                    if c in self._blockable:
                        s.add(c)
            return s

        self._bumps_left = lane_bumps(self.left_lane_cols)
        self._bumps_center = lane_bumps(self.center_lane_cols)
        self._bumps_right = lane_bumps(self.right_lane_cols)

        # MIX phase: optional alternating bumps (small) to reduce synchronized pathing.
        self._bumps_mix: Set[Coord] = set()
        if self.mix_extra_bumps:
            # Two small rectangles in the corridor, alternating left/right side.
            # Kept small so at least one wide corridor remains.
            rect1 = _rect(8, 9, y0, y0 + 1)
            rect2 = _rect(14, 15, y1 - 1, y1)
            for c in rect1 | rect2:
                if c in self._blockable:
                    self._bumps_mix.add(c)

        # Define phase patterns (which two lanes get bumps).
        self._phase_patterns: List[Tuple[Set[Coord], Set[Coord]]] = [
            (self._bumps_center, self._bumps_right),   # LEFT favored
            (self._bumps_left, self._bumps_center),    # RIGHT favored
            (self._bumps_left, self._bumps_right),     # CENTER favored
            (set(), set()),                             # MIX baseline (then add _bumps_mix)
        ]

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        # Rotate through 4 phases.
        phase = (t // self.phase_len) % 4
        bumps_a, bumps_b = self._phase_patterns[phase]

        blocked: Set[Coord] = set()
        blocked |= bumps_a
        blocked |= bumps_b

        if phase == 3:  # MIX
            blocked |= self._bumps_mix

        return blocked


# Meta-norm documentation
meta_norm = {
    "verbal_explanation": (
        "Load-Balancing Detours is a continuous route-shaping norm: it never fully closes "
        "the map with hard gates. Instead it places small, time-rotating 'speed-bump' obstacle "
        "clusters in the large empty corridor between the P spawn zone and the orchard. These "
        "bumps make certain lanes slightly less usable so A* paths tend to route different "
        "agents toward different regions (left / center / right) over time."
    ),
    "reasoning": (
        "This norm aims to maximize total apples collected over long horizons by reducing "
        "synchronized stampedes. When the 'best lane' shifts over time, nearest-apple paths "
        "become heterogeneous, distributing agents across patches. Distribution reduces the "
        "chance that any single patch neighborhood hits 0 nearby apples. Unlike seed banks, "
        "it does not protect apples. Unlike traffic lights, it does not create hard chokepoints."
    ),
    "code_with_placeholders": "See LoadBalancingDetoursNorm class implementation above.",
    "hyperparameters_for_this_environment": {
        "phase_len": 25,  # Faster rotation
        "lane_rows": [4, 13],  # Extended into apple area
        "left_lane_cols": [2, 3, 4, 5, 6, 7, 8, 9],  # Very wide overlapping lanes
        "center_lane_cols": [8, 9, 10, 11, 12, 13, 14, 15],
        "right_lane_cols": [14, 15, 16, 17, 18, 19, 20, 21],
        "bump_thickness": 10,  # Full 10 rows blocked
        "mix_extra_bumps": True,
        "phase_cycle": ["LEFT", "RIGHT", "CENTER", "MIX"]
    }
}
