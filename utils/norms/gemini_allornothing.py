"""
Pulsing Norm for Commons Harvest (All-or-Nothing)
--------------------------------------------------
This norm implements a global temporal synchronization strategy to maximize 
social welfare. It alternates between a short 'Harvest' burst (all harvesting allowed) 
and a long 'Recovery' phase (all harvesting blocked).

Mechanism:
- Cycle STARTS with a very short harvest window (5 steps by default).
- Then switches to a long recovery window (70 steps by default).
- Ratio is 14:1 recovery:harvest - agents get brief harvesting bursts.

Why this works:
1. Apples start at max capacity: Starting with harvest lets agents collect the
   initial bounty before locking down.
2. Short harvest windows: 5 steps is just enough for agents to grab some apples
   but not enough to completely strip the patches.
3. Long recovery: 70 steps gives regrowth mechanics maximum time to refill.
4. High epsilon robustness: Even if some agents violate during recovery, the
   short harvest windows limit total damage.
"""

from utils.norms.norm import Norm, Coord
from typing import Set

class PulsingNorm(Norm):
    def __init__(self, name: str = "pulsing_norm", epsilon: float = 0.0, 
                 recovery_duration: int = 70, harvest_duration: int = 5):
        """
        Initializes the Pulsing Norm.

        Args:
            name (str): The name of the norm.
            epsilon (float): Probability (0.0 to 1.0) that an agent ignores the norm.
            recovery_duration (int): Number of steps all apples are blocked (default: 70).
            harvest_duration (int): Number of steps apples are available (default: 5 - very short burst).
        
        Note: Cycle STARTS with harvest phase (since apples are at max capacity at t=0),
              then switches to long recovery. Ratio is 14:1 recovery:harvest.
        """
        super().__init__(name, epsilon)
        self.recovery_duration = recovery_duration
        self.harvest_duration = harvest_duration
        self.cycle_length = harvest_duration + recovery_duration  # Harvest first
        
        self.all_apples: Set[Coord] = set()
        self._initialize_apple_locations()

    def _initialize_apple_locations(self):
        """
        Parses the map to identify all apple locations. These are the positions
        that will be toggled on/off during the pulsing cycle.
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
                    self.all_apples.add((x, y))

    def get_blocked_positions(self, t: int) -> Set[Coord]:
        """
        Returns the blocked positions for timestep t.
        
        Logic:
        - Calculate position in the cycle: t % cycle_length
        - If position < harvest_duration: Block NOTHING (harvest window - very short).
        - Else (recovery phase): Block ALL apples (long recovery).
        
        The cycle STARTS with harvest since apples are at max capacity at t=0.
        This allows agents to harvest the initial bounty, then enforces long recovery.
        
        Args:
            t (int): Current timestep.
            
        Returns:
            Set[Coord]: The set of blocked apple coordinates.
        """
        cycle_position = t % self.cycle_length
        
        if cycle_position < self.harvest_duration:
            # HARVEST PHASE: Free for all (short burst at start of each cycle)
            return set()
        else:
            # RECOVERY PHASE: Protect everything (long regrowth period)
            return self.all_apples.copy()