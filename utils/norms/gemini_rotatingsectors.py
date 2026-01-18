"""
Rotating Sectors Norm for Commons Harvest
-----------------------------------------
This norm implements a temporal zoning strategy (crop rotation) to maximize 
social welfare. It divides the map into four quadrants and rotates access 
to them, forcing agents to allow patches to recover (lie fallow) between harvests.

Mechanism:
- The map is divided into 4 Sectors: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
- A 'period' hyperparameter defines how long a sector stays open.
- At any time t, 3 sectors are blocked, and 1 is open.
- The cycle ensures that a sector rests for (3 * period) steps before being reopened.

Why this works:
1. Prevents Total Depletion: Agents typically strip a patch in ~60 steps. 
   By rotating every 50 steps, we force agents to leave before the patch dies.
2. Synchronized Regrowth: While agents are in Sector A, Sectors B, C, and D 
   are strictly protected, allowing the density-dependent regrowth to maximize.
"""

from utils.norms.norm import Norm, Coord
from typing import Set, List, Tuple

class RotatingSectorsNorm(Norm):
    def __init__(self, name: str = "rotating_sectors", epsilon: float = 0.0, period: int = 50):
        """
        Initializes the Rotating Sectors Norm.

        Args:
            name (str): The name of the norm.
            epsilon (float): Probability (0.0 to 1.0) that an agent ignores the norm.
            period (int): The number of timesteps a sector remains open before switching.
                          Default is 50, based on the observation that patches die after 60 steps.
        """
        super().__init__(name, epsilon)
        self.period = period
        
        # Lists to store the coordinates of apples in each sector
        self.sectors: List[Set[Coord]] = [set(), set(), set(), set()]
        self.all_apples: Set[Coord] = set()
        
        # Map dimensions for sector splitting
        # Columns: 0-23 (Split at 12)
        # Rows: 0-17 (Split at 9)
        self.mid_x = 12
        self.mid_y = 9
        
        self._initialize_sectors()

    def _initialize_sectors(self):
        """
        Parses the map to identify apple locations and assigns them to 
        one of four geographical sectors.
        """
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

        for y, row_str in enumerate(ascii_map):
            for x, char in enumerate(row_str):
                if char == 'A':
                    coord = (x, y)
                    self.all_apples.add(coord)
                    
                    # Assign to sectors based on geometric position
                    # Sector 0: Top-Left
                    if x < self.mid_x and y < self.mid_y:
                        self.sectors[0].add(coord)
                    # Sector 1: Top-Right
                    elif x >= self.mid_x and y < self.mid_y:
                        self.sectors[1].add(coord)
                    # Sector 2: Bottom-Right
                    elif x >= self.mid_x and y >= self.mid_y:
                        self.sectors[2].add(coord)
                    # Sector 3: Bottom-Left
                    elif x < self.mid_x and y >= self.mid_y:
                        self.sectors[3].add(coord)

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Returns the blocked positions for timestep t.
        
        Logic:
        - Determine which sector is active based on time (t).
        - Block ALL apples that are NOT in the active sector.
        
        Args:
            t (int): Current timestep.
            
        Returns:
            Set[Coord]: The set of blocked apple coordinates.
        """
        # Determine the index of the currently open sector (0, 1, 2, or 3)
        # Cycle: 0 -> 1 -> 2 -> 3 -> 0 ...
        current_sector_index = (t // self.period) % 4
        
        # The active sector is the one allowed.
        # We must block everything else.
        active_sector_apples = self.sectors[current_sector_index]
        
        # Blocked = All Apples - Active Sector Apples
        # Note: We only return apple positions as blocked. 
        # Walls and empty space are handled by the agent's physics/logic.
        blocked_positions = self.all_apples - active_sector_apples
        
        return blocked_positions