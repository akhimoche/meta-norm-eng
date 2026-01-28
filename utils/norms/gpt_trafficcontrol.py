"""
Traffic-Light Gates Norm for Commons Harvest.

A throttling + partitioning norm that shapes when and where agents can traverse
key high-traffic corridors using timed gates and spawn pens.
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
                spaces.add(c)  # treat Q tiles as walkable space for movement planning
            elif ch == "P":
                p_spawns.add(c)
                spaces.add(c)  # treat P tiles as walkable space for movement planning
            else:
                # Any unknown symbol treated as walkable space.
                spaces.add(c)

    return {
        "walls": walls,
        "apples": apples,
        "spaces": spaces,
        "q_spawns": q_spawns,
        "p_spawns": p_spawns,
        "width": {(w, 0)},   # tiny hack: store w/h without changing return type
        "height": {(h, 0)},
    }


def _wh_from_parsed(parsed: Dict[str, Set[Coord]]) -> Tuple[int, int]:
    # Extract width/height from the tiny hack store.
    w = next(iter(parsed["width"]))[0]
    h = next(iter(parsed["height"]))[0]
    return w, h


def _n4(c: Coord) -> List[Coord]:
    x, y = c
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


class TrafficLightGatesNorm(Norm):
    """
    TrafficLightGatesNorm (Commons Harvest Open)

    Behaviour
    ----------
    A stateless, time-scheduled "traffic light" system that restricts movement through
    artificial chokepoints (gate-lines + divider + Q pens). It throttles early rush,
    partitions agents left/right over time, and introduces periodic rest windows.

    Key components
    --------------
    1) South gate-line: a horizontal wall of blocked *space tiles* just above the P spawn zone,
       with a few doorway cells that open/close over time.
    2) Mid divider: a vertical wall of blocked *space tiles* around the map's midline (col ~ 12),
       with a timed door for occasional rebalancing.
    3) Q pens: when "closed", block the 4-neighborhood around each Q spawn (space tiles only),
       leaving Q agents unable to leave; when "open", unblock exactly one exit cell.

    Schedule (cycle repeats)
    ------------------------
    - LEFT window: left doorway open; left Q exit open; right Q penned; divider closed.
    - RIGHT window: right doorway open; right Q exit open; left Q penned; divider closed.
    - REST window: all doorways closed; both Q penned; divider closed.
    - MIX window: center doorway + divider door open; both Q exits open.

    Hyperparameters
    ---------------
    warmup_steps : int
        Initial steps where both side doors are open and both Q exits open (to avoid accidental deadlocks).
    cycle_len : int
        Length of the repeating schedule cycle.
    window_len : int
        Length of each of the 4 windows inside the cycle. Must satisfy 4*window_len == cycle_len.
    left_door_cols / right_door_cols / center_door_cols : Tuple[int, ...]
        Doorway columns on the south gate-line row.
    divider_col : int
        Column index for the mid divider.
    divider_door_rows : Tuple[int, ...]
        Rows where a divider door exists (unblocked during MIX and warmup).

    Notes
    -----
    - This norm blocks only *space/spawn tiles*, never apple tiles, keeping it conceptually distinct
      from seed-bank norms.
    - Fully self-contained: no external state required.
    """

    def __init__(
        self,
        epsilon: float = 0.0,
        *,
        warmup_steps: int = 10,  # Shorter warmup = gates activate sooner
        window_len: int = 30,  # Longer windows = more time spent in REST
        left_door_cols: Tuple[int, ...] = (5,),  # Single-tile doors = tighter chokepoint
        right_door_cols: Tuple[int, ...] = (18,),  # Single-tile doors
        center_door_cols: Tuple[int, ...] = (11,),  # Single-tile door
        divider_col: int = 12,
        divider_door_rows: Tuple[int, ...] = (),  # NO divider doors = permanent wall
    ):
        super().__init__("traffic_light_gates", epsilon)

        parsed = _parse_ascii_map(ASCII_MAP)
        w, h = _wh_from_parsed(parsed)

        self._walls = parsed["walls"]
        self._spaces = parsed["spaces"]
        self._apples = parsed["apples"]
        self._q_spawns = parsed["q_spawns"]
        self._p_spawns = parsed["p_spawns"]
        self._w = w
        self._h = h

        self.warmup_steps = int(warmup_steps)
        self.window_len = int(window_len)
        self.cycle_len = 4 * self.window_len

        self.left_door_cols = tuple(int(c) for c in left_door_cols)
        self.right_door_cols = tuple(int(c) for c in right_door_cols)
        self.center_door_cols = tuple(int(c) for c in center_door_cols)

        self.divider_col = int(divider_col)
        self.divider_door_rows = tuple(int(r) for r in divider_door_rows)

        # Blockable tiles are walkable non-apple spaces, excluding spawn tiles (avoid blocking start positions).
        self._blockable: Set[Coord] = set(self._spaces) - set(self._apples) - set(self._q_spawns) - set(self._p_spawns)

        # Define the south gate-line row: moved up to row 8 (inside the apple area)
        # This creates a hard barrier agents must pass through to reach upper patches
        self.south_gate_row = 8

        # South gate-line base barrier: all blockable tiles across that row (usually columns 1..w-2).
        self._south_barrier_base: Set[Coord] = {
            (x, self.south_gate_row)
            for x in range(w)
            if (x, self.south_gate_row) in self._blockable
        }

        # Divider base barrier: vertical line down divider_col, from row 1 up to the south gate row.
        self._divider_base: Set[Coord] = {
            (self.divider_col, y)
            for y in range(1, self.south_gate_row + 1)
            if (self.divider_col, y) in self._blockable
        }

        # Divider door cells (subset of divider_base) at specified rows.
        self._divider_doors: Set[Coord] = {
            (self.divider_col, y)
            for y in self.divider_door_rows
            if (self.divider_col, y) in self._divider_base
        }

        # South door cells (subset of south_barrier_base).
        self._south_left_doors: Set[Coord] = {(c, self.south_gate_row) for c in self.left_door_cols if (c, self.south_gate_row) in self._south_barrier_base}
        self._south_right_doors: Set[Coord] = {(c, self.south_gate_row) for c in self.right_door_cols if (c, self.south_gate_row) in self._south_barrier_base}
        self._south_center_doors: Set[Coord] = {(c, self.south_gate_row) for c in self.center_door_cols if (c, self.south_gate_row) in self._south_barrier_base}

        # Q pens: for each Q, block its 4-neighborhood (blockable tiles only). Door = prefer south neighbor else any neighbor.
        self._q_pen_cells: Dict[Coord, Set[Coord]] = {}
        self._q_door_cell: Dict[Coord, Coord] = {}
        for q in self._q_spawns:
            pen = {n for n in _n4(q) if n in self._blockable}
            self._q_pen_cells[q] = pen

            # Choose a door (an exit cell to unblock when releasing).
            preferred = (q[0], q[1] + 1)
            if preferred in pen:
                door = preferred
            else:
                door = next(iter(pen)) if pen else q  # if no pen cells exist, degenerate safely
            self._q_door_cell[q] = door

        # Identify left vs right Qs by divider_col.
        self._left_qs = {q for q in self._q_spawns if q[0] < self.divider_col}
        self._right_qs = set(self._q_spawns) - self._left_qs

    def _phase(self, t: int) -> int:
        """
        Returns:
            -1 for warmup,
             0 for LEFT window,
             1 for RIGHT window,
             2 for REST window,
             3 for MIX window.
        """
        if t < self.warmup_steps:
            return -1
        cyc = (t - self.warmup_steps) % self.cycle_len
        return cyc // self.window_len  # 0..3

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        phase = self._phase(t)

        blocked: Set[Coord] = set()

        # Base barriers always applied; doors are removed when "open".
        blocked |= self._south_barrier_base
        blocked |= self._divider_base

        # Q pens always applied; door cells removed when the Q is "released".
        for q, pen_cells in self._q_pen_cells.items():
            blocked |= pen_cells

        # Decide open doors by phase.
        open_south_doors: Set[Coord] = set()
        divider_open: bool = False
        release_left_q: bool = False
        release_right_q: bool = False

        if phase == -1:
            # Warmup: allow movement, but still funnel through limited doors.
            open_south_doors |= self._south_left_doors | self._south_right_doors
            divider_open = True
            release_left_q = True
            release_right_q = True
        elif phase == 0:  # LEFT
            open_south_doors |= self._south_left_doors
            divider_open = False
            release_left_q = True
            release_right_q = False
        elif phase == 1:  # RIGHT
            open_south_doors |= self._south_right_doors
            divider_open = False
            release_left_q = False
            release_right_q = True
        elif phase == 2:  # REST
            open_south_doors |= set()  # none
            divider_open = False
            release_left_q = False
            release_right_q = False
        else:  # MIX
            open_south_doors |= self._south_center_doors | self._south_left_doors | self._south_right_doors
            divider_open = True
            release_left_q = True
            release_right_q = True

        # Apply south doors.
        blocked -= open_south_doors

        # Apply divider doors.
        if divider_open:
            blocked -= self._divider_doors
        else:
            blocked |= self._divider_doors  # (already in divider_base; explicit for clarity)

        # Release Qs by unblocking exactly one exit tile per Q.
        if release_left_q:
            for q in self._left_qs:
                blocked.discard(self._q_door_cell[q])
        if release_right_q:
            for q in self._right_qs:
                blocked.discard(self._q_door_cell[q])

        return blocked


# Meta-norm documentation
meta_norm = {
    "verbal_explanation": (
        "Traffic-Light Gates is a throttling + partitioning norm that shapes when and where "
        "agents can traverse key high-traffic corridors. It builds artificial gates above the "
        "spawn zone and a vertical divider down the map's midline. The gates cycle through "
        "LEFT, RIGHT, REST, and MIX windows, controlling which side agents can access. "
        "Q-spawn agents are penned and released on schedule. This does not protect apples "
        "directly - it constrains movement chokepoints and adds cooldown time."
    ),
    "reasoning": (
        "This norm increases total apples by: preventing the earliest high-density rush "
        "(especially from Q spawns), splitting agents across left/right regions over time, "
        "and introducing enforced rest periods so apples are less likely to reach the "
        "0-neighbor regrowth collapse. Even at high epsilon, intermittent compliance creates "
        "'sticky' macro effects: agents routed into one side tend to keep harvesting there."
    ),
    "code_with_placeholders": "See TrafficLightGatesNorm class implementation above.",
    "hyperparameters_for_this_environment": {
        "warmup_steps": 10,  # Quick warmup
        "window_len": 30,  # Longer windows
        "left_door_cols": [5],  # Single-tile doors (tighter chokepoints)
        "right_door_cols": [18],
        "center_door_cols": [11],
        "divider_col": 12,
        "divider_door_rows": [],  # Permanent wall - no doors
        "south_gate_row": 8,  # Inside apple area
        "schedule_summary": {
            "cycle_len": "4 * window_len = 120",
            "windows": ["LEFT", "RIGHT", "REST", "MIX"]
        }
    }
}
