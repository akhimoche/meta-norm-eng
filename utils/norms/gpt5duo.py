from typing import Set, Tuple, Dict, Callable
from utils.norms.norm import Norm, Coord

class PulseRotationRefugiaOpen(Norm):
    """
    Pulse–Rotation with Permanent Refugia for Commons Harvest Open (24×18).
    
    Core idea:
    - Permanently close a small, well-dispersed share of cells (refugia) so seed stock persists.
    - Partition the grid into modular tiles (K=12). Each timestep, only a narrow window of tiles (W=2)
      is open; the window sweeps deterministically over time (CYCLE=144) with a small time-only jitter.
    - Add a light micro-rotation layer (K=3, W=2) so even within an open coarse tile, only ~2/3 of
      its area is accessible; this creates longer cooldowns at multiple spatial scales.
    - Thin openings within open tiles with a deterministic “dither” pattern (~75% of those cells open),
      which caps instantaneous pressure without any agent state.
    - Add a 1-cell guard band ONLY around refugia (not around all closed tiles) to mitigate edge
      depletion without over-closing the map.
      
    Resulting invariants (by construction):
      • ≥ 12% of cells are always protected as refugia (plus a 1-cell buffer around them).
      • At most ≈ 8.3% of the grid is open at any timestep:
          (W/K) * (W_micro/K_micro) * dither ≈ (2/12) * (2/3) * 0.75 ≈ 0.0833.
      • Each coarse tile’s typical closed duration ≈ CYCLE * (K-W)/K = 144 * 10/12 = 120 steps,
        long relative to depletion (~60) so patches can rebuild before reopening.
    """
    # ---------------------- Static grid facts ----------------------
    GRID_W: int = 24
    GRID_H: int = 18

    # ---------------------- Tuned hyperparameters ----------------------
    # Coarse rotation (primary gate)
    K: int = 12                # number of rotation tiles
    W: int = 2                 # open window width in tile-index space
    CYCLE: int = 144           # timesteps for a full sweep
    TILE_HASH: Tuple[int, int, int] = (5, 7, 11)  # co-prime with K to avoid bias

    # Micro rotation (nested gate for robustness across patch sizes)
    MS_LAYER: Dict[str, object] = {
        "K": 3,
        "W": 2,
        "CYCLE": 60,
        "TILE_HASH": (3, 1, 2)  # different coefficients than coarse layer
    }

    # Permanent refugia (always closed)
    REFUGIA_FRACTION: float = 0.12
    REFUGIA_SALT: int = 4242

    # Pressure throttling inside open tiles
    OPEN_PATTERN: str = "dither"  # ~75% of cells in open tiles are available
    REFUGIA_EDGE_BUFFER: int = 1  # 1-cell guard band around refugia only

    def __init__(self, epsilon: float = 0.0):
        super().__init__("pulse_rotation_refugia_open", epsilon)

    # ---------------------- Deterministic helpers ----------------------
    @staticmethod
    def _lcg_hash(x: int, y: int, salt: int) -> int:
        # Cheap, deterministic 32-bit LCG hash
        return (1103515245 * (x + 31 * y + 131 * salt) + 12345) & 0x7fffffff

    @staticmethod
    def _tile_index(x: int, y: int, K: int, a: int, b: int, c: int) -> int:
        # Map (x,y) -> {0..K-1}; a,b should be co-prime with K to avoid spatial bias
        return ((a * x + b * y + c) % K + K) % K

    @staticmethod
    def _slow_spin_jitter(t: int, J: int) -> int:
        # time-only phase jitter; rotates by 1 every J steps
        return (t // J)

    # ---------------------- Spatial predicates ----------------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.GRID_W and 0 <= y < self.GRID_H

    def _in_refugia(self, x: int, y: int) -> bool:
        if self.REFUGIA_FRACTION <= 0.0:
            return False
        h = self._lcg_hash(x, y, self.REFUGIA_SALT)
        # Normalize to [0,1)
        return (h / float(0x80000000)) < self.REFUGIA_FRACTION

    def _open_in_coarse_tile(self, x: int, y: int, t: int) -> bool:
        K, W, period = self.K, self.W, self.CYCLE
        a, b, c = self.TILE_HASH
        base = (t % period) * K // period  # linear sweep over tiles per cycle
        phase = (base + (self._slow_spin_jitter(t, J=9))) % K
        idx = self._tile_index(x, y, K, a, b, c)
        return ((idx - phase) % K) < W

    def _open_in_micro_tile(self, x: int, y: int, t: int) -> bool:
        ms = self.MS_LAYER
        K, W, period = int(ms["K"]), int(ms["W"]), int(ms["CYCLE"])
        a, b, c = tuple(ms["TILE_HASH"])
        base = (t % period) * K // period
        phase = (base + (self._slow_spin_jitter(t, J=11))) % K
        idx = self._tile_index(x, y, K, a, b, c)
        return ((idx - phase) % K) < W

    @staticmethod
    def _thinning_pattern(x: int, y: int, t: int, mode: str) -> bool:
        # - 'exact': open everything in the open tile
        # - 'stripe': open ~50% via parity
        # - 'dither': open ~75% via simple modular rule
        if mode == "exact":
            return True
        elif mode == "stripe":
            return ((x + t) & 1) == 0
        elif mode == "dither":
            return ((x * 3 + y * 5 + t) % 4) != 0
        else:
            # default: permissive
            return True

    # ---------------------- Core decision ----------------------
    def _is_open(self, x: int, y: int, t: int) -> bool:
        # 1) permanent refugia are always closed
        if self._in_refugia(x, y):
            return False
        # 2) both rotation layers must approve
        if not self._open_in_coarse_tile(x, y, t):
            return False
        if not self._open_in_micro_tile(x, y, t):
            return False
        # 3) thinning inside open tiles
        if not self._thinning_pattern(x, y, t, self.OPEN_PATTERN):
            return False
        return True

    # ---------------------- Public API ----------------------
    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Returns blocked positions at timestep t.
        Implements Pulse–Rotation with Permanent Refugia specialized for Commons Harvest Open.
        """
        blocked: Set[Coord] = set()

        # First pass: mark refugia (permanent)
        refugia: Set[Coord] = set()
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self._in_refugia(x, y):
                    refugia.add((x, y))  # type: ignore[typeddict-item]

        # Add guard band ONLY around refugia to prevent edge depletion
        if self.REFUGIA_EDGE_BUFFER > 0:
            buf = self.REFUGIA_EDGE_BUFFER
            expanded_refugia: Set[Coord] = set(refugia)
            for (rx, ry) in list(refugia):
                for dx in range(-buf, buf + 1):
                    for dy in range(-buf, buf + 1):
                        xx, yy = rx + dx, ry + dy
                        if self._in_bounds(xx, yy):
                            expanded_refugia.add((xx, yy))  # type: ignore
            refugia = expanded_refugia

        # Second pass: apply rotation/thinning; anything not open is blocked
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                coord: Coord = (x, y)  # type: ignore
                # Always block refugia (and its guard band)
                if coord in refugia:
                    blocked.add(coord)
                    continue
                # Outside bounds (shouldn't happen here), treat as blocked
                if not self._in_bounds(x, y):
                    blocked.add(coord)
                    continue
                # Rotation + thinning decide openness
                if not self._is_open(x, y, t):
                    blocked.add(coord)

        return blocked

# Expose for dynamic loading
NormClass = PulseRotationRefugiaOpen
