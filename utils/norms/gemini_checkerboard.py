"""
Lattice Preservation Norm for Commons Harvest
---------------------------------------------
This norm implements a static checkerboard restriction pattern to maximize 
social welfare (total harvest) by optimizing the density-dependent regrowth 
mechanics of the environment.

Mechanism:
- The norm parses the environment map to locate all apple patches.
- It calculates a "parity" for every coordinate (x + y).
- It blocks all apples where (x + y) is EVEN.
- This creates a lattice where every "Open" (ODD) apple is surrounded by 
  4 "Blocked" (EVEN) apples.

Why this works:
1. Max Regrowth: Harvestable apples will have up to 4 neighbors (the blocked ones),
   ensuring they hit the max regrowth probability (2.5%) immediately after being eaten.
2. Robustness: Unlike norms that protect a single central "seed", this distributes 
   seeds everywhere. Even with high epsilon (agents ignoring the norm), it is 
   statistically difficult to wipe out the entire lattice structure.
"""

from utils.norms.norm import Norm, Coord
from typing import Set

class LatticePreservationNorm(Norm):
    def __init__(self, name: str = "lattice_preservation", epsilon: float = 0.0):
        """
        Initializes the Lattice Preservation Norm.
        
        Args:
            name (str): The name of the norm.
            epsilon (float): Probability (0.0 to 1.0) that an agent ignores the norm 
                             at any given timestep.
        """
        super().__init__(name, epsilon)
        self.blocked_positions: Set[Coord] = set()
        self._initialize_lattice()

    def _initialize_lattice(self):
        """
        Parses the ASCII map to identify all apple locations ('A') and 
        permanently blocks those that fall on 'even' checkerboard squares.
        """
        # The exact map layout provided in the environment description
        ascii_map = [
            "WWWWWWWWWWWWWWWWWWWWWWWW",
            "WAAA    A      A    AAAW",
            "WAA    AAA    AAA    AAW",
            "WA    AAAAA  AAAAA    AW",
            "W      AAA    AAA      W",
            "W       A      A       W",
            "W  A                A  W",
            "W AAA  Q        Q  AAA W",
            "WAAAAA            AAAAAW",
            "W AAA              AAA W",
            "W  A                A  W",
            "W                      W",
            "W                      W",
            "W                      W",
            "W  PPPPPPPPPPPPPPPPPP  W",
            "W PPPPPPPPPPPPPPPPPPPP W",
            "WPPPPPPPPPPPPPPPPPPPPPPW",
            "WWWWWWWWWWWWWWWWWWWWWWWW"
        ]

        # Iterate over the map to find apples
        for y, row_str in enumerate(ascii_map):
            for x, char in enumerate(row_str):
                if char == 'A':
                    # Check parity: Block if (x + y) is even.
                    # This ensures a perfect checkerboard.
                    # We block 'even' specifically because in many dense patches 
                    # on this specific map, blocking evens retains a slightly 
                    # higher number of seeds than blocking odds, offering better stability.
                    if (x + y) % 2 == 0:
                        self.blocked_positions.add((x, y))

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Returns the set of blocked coordinates for the current timestep.
        
        For the Lattice norm, the blocked positions are static and do not change 
        over time, providing a consistent structure for agents to learn around.
        
        Args:
            t (int): Current timestep.
            
        Returns:
            Set[Coord]: A set of (x, y) tuples representing blocked cells.
        """
        return self.blocked_positions