# Section 0: Standard library imports
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Set, Tuple

from utils.norms.norm import Norm, Coord


class CoreSanctuaryNorm(Norm):
    """Core-Sanctuary norm for Commons Harvest (Open layout).

    Summary
    -------
    Permanently protects the *inner* cells of each apple patch so agents can only harvest the outer rim.
    This maintains local apple density (3+ within respawn radius) and prevents permanent patch collapse.

    How it works
    ------------
    - Parses the fixed 24x18 ASCII map.
    - Finds connected components of 'A' tiles (apple patches).
    - For each patch, computes an 'inward depth' by repeatedly peeling boundary tiles.
      * Large patches (max_depth >= 2): block tiles with depth >= `SANCTUARY_DEPTH_THRESHOLD`.
      * Small patches (no interior): block `SMALL_PATCH_SEEDS` centrally-scored tiles.
    - Optionally adds a tiny moat of adjacent floor tiles around small-patch sanctuaries.

    Why it is robust to epsilon
    ---------------------------
    Intermittent noncompliance does not immediately remove the norm's benefit because the protected reserve
    is spatially distributed (not just a few hand-picked apples), preserving regrowth-supporting density in
    many neighborhoods. Under perfect compliance (epsilon=0.0), the reserve is never harvested.

    Hyperparameters
    ---------------
    SANCTUARY_DEPTH_THRESHOLD : int
        For large patches, tiles with depth >= this value are blocked (protected). Default: 1.
    SMALL_PATCH_SEEDS : int
        For small patches (no interior depth), how many tiles to protect. Default: 3.
    SMALL_PATCH_MOAT_RADIUS : int
        Manhattan radius of extra *floor* tiles to block around small-patch sanctuaries. Default: 1.
    """

    # ---- Tuneables (safe defaults for this exact Open map) ----
    SANCTUARY_DEPTH_THRESHOLD: int = 1
    SMALL_PATCH_SEEDS: int = 3
    SMALL_PATCH_MOAT_RADIUS: int = 1

    ASCII_MAP: str = """
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

    def __init__(self, epsilon: float = 0.0):
        super().__init__("core_sanctuary", epsilon)
        (
            self._walls,
            self._floors,
            self._apples,
            self._spawns,
        ) = self._parse_map(self.ASCII_MAP)

        patches = self._connected_components(self._apples)

        blocked: Set[Coord] = set()

        for patch in patches:
            depth = self._inward_depth(patch)
            max_depth = max(depth.values()) if depth else 0

            if max_depth >= 2:
                # Large patch: protect all inner layers depth >= threshold.
                sanctuary = {c for c, d in depth.items() if d >= self.SANCTUARY_DEPTH_THRESHOLD}
            else:
                # Small patch: no interior; pick a central subset to protect.
                sanctuary = self._pick_central_seeds(patch, k=min(self.SMALL_PATCH_SEEDS, len(patch)))

            blocked |= sanctuary

            # Optional moat: only applied around small-patch sanctuaries (kept tiny to avoid over-blocking).
            if max_depth < 2 and self.SMALL_PATCH_MOAT_RADIUS > 0:
                blocked |= self._moat_floor_around(
                    sanctuary,
                    radius=self.SMALL_PATCH_MOAT_RADIUS,
                    floors=self._floors,
                    forbidden=self._spawns,
                )

        # Never block spawn tiles (safety).
        blocked -= self._spawns
        self._static_blocked: Set[Coord] = blocked

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Return the protected core positions (constant across all timesteps).
        
        Args:
            t: Current simulation timestep (unused for static norm)
            
        Returns:
            Set of blocked coordinates
        """
        # Static norm: no external state required.
        return set(self._static_blocked)

    # ----------------------- Helpers -----------------------
    @staticmethod
    def _parse_map(ascii_map: str) -> Tuple[Set[Coord], Set[Coord], Set[Coord], Set[Coord]]:
        lines = [ln for ln in ascii_map.splitlines() if ln.strip() != ""]
        walls: Set[Coord] = set()
        floors: Set[Coord] = set()
        apples: Set[Coord] = set()
        spawns: Set[Coord] = set()

        for y, row in enumerate(lines):
            for x, ch in enumerate(row):
                c: Coord = (x, y)
                if ch == "W":
                    walls.add(c)
                else:
                    floors.add(c)
                    if ch == "A":
                        apples.add(c)
                    elif ch in {"Q", "P"}:
                        spawns.add(c)
        return walls, floors, apples, spawns

    @staticmethod
    def _neighbors4(c: Coord) -> Iterable[Coord]:
        x, y = c
        yield (x + 1, y)
        yield (x - 1, y)
        yield (x, y + 1)
        yield (x, y - 1)

    def _connected_components(self, cells: Set[Coord]) -> List[Set[Coord]]:
        seen: Set[Coord] = set()
        comps: List[Set[Coord]] = []
        for start in cells:
            if start in seen:
                continue
            q = deque([start])
            seen.add(start)
            comp: Set[Coord] = {start}
            while q:
                cur = q.popleft()
                for nb in self._neighbors4(cur):
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        comp.add(nb)
                        q.append(nb)
            comps.append(comp)
        # Deterministic ordering (nice for debugging): by centroid then size.
        def centroid(comp: Set[Coord]) -> Tuple[float, float]:
            xs = [c[0] for c in comp]
            ys = [c[1] for c in comp]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        comps.sort(key=lambda c: (centroid(c)[1], centroid(c)[0], len(c)))
        return comps

    def _inward_depth(self, patch: Set[Coord]) -> Dict[Coord, int]:
        """Peel boundary layers to compute depth inward.

        Depth 0: boundary tiles (touching outside of patch via 4-neighborhood).
        Depth 1: after removing depth-0, next boundary, etc.
        """
        remaining: Set[Coord] = set(patch)
        depth: Dict[Coord, int] = {}
        d = 0
        while remaining:
            boundary: Set[Coord] = set()
            for c in remaining:
                # boundary if any 4-neighbor is not in remaining
                if any(nb not in remaining for nb in self._neighbors4(c)):
                    boundary.add(c)
            for c in boundary:
                depth[c] = d
            remaining -= boundary
            d += 1
            # Safety: prevent infinite loops (should never happen)
            if d > 50:
                break
        return depth

    @staticmethod
    def _pick_central_seeds(patch: Set[Coord], k: int) -> Set[Coord]:
        """Pick k tiles with highest local density (centrality) inside this patch."""
        if k <= 0:
            return set()
        patch_list = list(patch)
        # Score = number of patch tiles within Euclidean distance <= 2.0
        scored: List[Tuple[int, int, int, Coord]] = []
        for c in patch_list:
            cx, cy = c
            count = 0
            for o in patch_list:
                ox, oy = o
                if (cx - ox) ** 2 + (cy - oy) ** 2 <= 4:
                    count += 1
            # Tie-breakers: prefer more central (by mean position), then stable ordering.
            scored.append((count, -abs(cx), -abs(cy), c))
        scored.sort(reverse=True)
        return {c for *_, c in scored[:k]}

    def _moat_floor_around(
        self,
        sanctuary: Set[Coord],
        radius: int,
        floors: Set[Coord],
        forbidden: Set[Coord],
    ) -> Set[Coord]:
        """Block a small ring of floor tiles around sanctuary tiles (used only for small patches)."""
        if radius <= 0:
            return set()
        moat: Set[Coord] = set()
        for sx, sy in sanctuary:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue
                    c: Coord = (sx + dx, sy + dy)
                    if c in floors and c not in self._apples and c not in forbidden:
                        moat.add(c)
        return moat


# Meta-norm documentation following required format
meta_norm = {
  "verbal_explanation": (
        "I'm proposing 5 conceptually different 'plug-and-play' norm families for Commons Harvest:\n\n"
        "1) **Core-Sanctuary (patch-internal reserves)**: Permanently block a patch's inner cells so agents can only harvest the outer rim. "
        "This leaves a stable 'seed stock' of uneaten apples that keeps local neighborhood density high, sustaining regrowth.\n"
    "2) **Staggered Patch Rest Windows (time-based closures)**: Periodically seal entire patches (by blocking their boundary-adjacent tiles) "
    "so patches alternate between harvesting and recovery phases.\n"
    "3) **Rotating Sector Access (time-sliced zoning)**: Divide the map into sectors and only allow movement into some sectors at a time, "
    "forcing dispersed foraging and preventing dogpiling one patch.\n"
    "4) **Chokepoint Toll Booths (movement friction)**: Block a small set of high-traffic corridor cells so intermittent compliance causes "
    "replanning/oscillation and slows the rush-to-apple dynamics even at high epsilon.\n"
    "5) **Parity / Checkerboard Speed-Limit (global throttling)**: Block a structured subset of traversable cells (e.g., parity classes) "
    "to reduce effective movement speed and spread harvesting pressure over time.\n\n"
    "Random pick (for code): **#1 Core-Sanctuary**.\n\n"
        "**What the implemented norm does:** It detects the 6 apple patches directly from the ASCII map, computes each patch's 'inward depth' "
    "(peeling boundary layers), and blocks the *inner* cells of each large patch (depth ≥ 1). For the two small corner patches (size 6), "
        "it blocks a small central subset. Optionally it also adds a tiny 'moat' of adjacent floor tiles around those small sanctuaries.\n\n"
    "**Why this helps welfare:** With density-dependent regrowth, keeping a stable inner reserve means most regrowth neighborhoods retain "
    "3+ apples, unlocking the 2.5% per-step regrowth probability and preventing permanent barrenness. Agents still harvest the outer ring, "
    "so collection remains efficient, but the game no longer collapses by ~t=60."
  ),
  "reasoning": (
        "Commons Harvest's collapse happens because agents 'strip-mine' patches to zero local density; once neighborhoods lose apples, regrowth "
    "probability drops to ~0. A norm that *guarantees persistent local apple density* can preserve long-run productivity.\n\n"
    "Core-Sanctuary does that without any live state: it uses only the fixed map geometry. For each 13-tile patch, blocking the inner layer "
    "(depth ≥ 1) leaves 8 boundary tiles harvestable and 5 inner tiles protected. The protected inner tiles act as a permanent seed bank.\n\n"
    "Robustness to epsilon: even if agents intermittently ignore obstacles, the norm still raises expected welfare because (i) inner reserves "
        "are harder to fully eliminate than 'block 3 apples' rules (the reserve is distributed across each patch), and (ii) any compliance keeps "
    "some apples permanently unharvested, preserving regrowth capacity. In your experiments (epsilon=0.0), it should strongly prevent early "
    "barren patches while allowing sustained harvesting across 500 steps.\n\n"
    "Integration: agents already union physical obstacles with `norm.get_blocked_positions(t)`; blocking specific `Coord`s is exactly what "
    "A* uses to avoid stepping onto those cells. No other file changes needed."
  ),
    "code_with_placeholders": "See CoreSanctuaryNorm class implementation above.",
  "hyperparameters_for_this_environment": {
    "norm_name_passed_to_super": "core_sanctuary",
    "SANCTUARY_DEPTH_THRESHOLD": 1,
    "SMALL_PATCH_SEEDS": 3,
    "SMALL_PATCH_MOAT_RADIUS": 1,
    "notes": (
      "For the Open map, this yields: large patches protect 5 inner tiles each (4 at depth1 + 1 at depth2), "
      "small corner patches protect 3 tiles each (+ tiny moat). This leaves 38/64 apple tiles harvestable "
      "and should prevent early barren collapse under epsilon=0.0 while remaining reasonably robust to partial violations."
    )
  }
}
