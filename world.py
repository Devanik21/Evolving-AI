"""
world.py — Project A.L.I.V.E. NEXUS
The World & Environment Engine

Maze Generation Algorithms:
  • Recursive Backtracker (DFS) — long winding corridors, few dead-ends
  • Prim's Algorithm           — branching structure, many dead-ends
  • Wilson's Algorithm         — uniform spanning tree, unbiased
  • Hybrid                     — combines all three for fractal complexity

Environment Features:
  • Fog-of-war with vision radius
  • Dynamic obstacles (homing traps)
  • Teleport portals
  • Reward shaping (potential-based)
  • 9-cell local vision state encoding
  • A* shortest path solver (for comparison)
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Optional, Set
import heapq


# ============================================================
#  CELL TYPES
# ============================================================
WALL    = 1
PATH    = 0
AGENT   = 2
TARGET  = 3
PORTAL  = 4
TRAP    = 5
VISITED = 6

# ============================================================
#  MAZE GENERATOR
# ============================================================
class MazeGenerator:
    """
    Generates perfect mazes (exactly one path between any two cells)
    using three distinct algorithms, each producing unique topologies.
    """

    @staticmethod
    def _make_wall_grid(h: int, w: int) -> np.ndarray:
        """Start with all walls; cells at even (r,c) are potential passages."""
        grid = np.ones((h, w), dtype=np.int8)
        return grid

    # ----------------------------------------------------------
    @staticmethod
    def backtracker(h: int, w: int, seed: int = None) -> np.ndarray:
        """
        Recursive Backtracker (DFS).
        With added Straight-Bias: produces longer hallways and fewer turns, 
        reducing branching complexity while maintaining a perfect maze topology.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        ch, cw = (h - 1) // 2, (w - 1) // 2
        grid = np.ones((h, w), dtype=np.int8)
        straight_bias = 0.65  # 65% chance to prioritize continuing straight

        def carve(cr, cc, last_dr=0, last_dc=0):
            grid[2 * cr + 1, 2 * cc + 1] = PATH
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            rng.shuffle(directions)
            
            # Bias toward maintaining the same direction to reduce winding complexity
            if (last_dr, last_dc) in directions and rng.random() < straight_bias:
                directions.remove((last_dr, last_dc))
                directions.insert(0, (last_dr, last_dc))

            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < ch and 0 <= nc < cw and grid[2*nr+1, 2*nc+1] == WALL:
                    grid[2*cr+1 + dr, 2*cc+1 + dc] = PATH
                    carve(nr, nc, dr, dc)

        import sys
        sys.setrecursionlimit(10_000)
        carve(0, 0)
        return grid

    # ----------------------------------------------------------
    @staticmethod
    def prim(h: int, w: int, seed: int = None) -> np.ndarray:
        """
        Randomized Prim's Algorithm.
        Produces highly branching mazes with many dead-ends.
        Trains the agent to backtrack and make decisions at junctions.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        ch, cw = (h - 1) // 2, (w - 1) // 2
        grid = np.ones((h, w), dtype=np.int8)
        visited = set()

        def in_bounds(r, c):
            return 0 <= r < ch and 0 <= c < cw

        def neighbors(r, c):
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if in_bounds(nr, nc):
                    yield nr, nc

        start = (rng.randrange(ch), rng.randrange(cw))
        visited.add(start)
        grid[2*start[0]+1, 2*start[1]+1] = PATH
        frontier = list(neighbors(*start))

        while frontier:
            idx = rng.randrange(len(frontier))
            r, c = frontier.pop(idx)
            if (r, c) in visited:
                continue
            # Connect to a random already-visited neighbor
            connected = [(nr, nc) for nr, nc in neighbors(r, c) if (nr, nc) in visited]
            if connected:
                nr, nc = rng.choice(connected)
                # Carve passage
                grid[2*r+1, 2*c+1] = PATH
                grid[r+nr+1, c+nc+1] = PATH
                visited.add((r, c))
                for nb in neighbors(r, c):
                    if nb not in visited:
                        frontier.append(nb)

        return grid

    # ----------------------------------------------------------
    @staticmethod
    def wilson(h: int, w: int, seed: int = None) -> np.ndarray:
        """
        Wilson's Algorithm (Loop-Erased Random Walk).
        Produces an unbiased uniform spanning tree — mathematically perfect.
        Topologically very different from DFS or Prim.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        ch, cw = (h - 1) // 2, (w - 1) // 2
        grid = np.ones((h, w), dtype=np.int8)
        in_maze: Set[Tuple[int,int]] = set()

        def in_bounds(r, c):
            return 0 <= r < ch and 0 <= c < cw

        def nb(r, c):
            return [(r+dr, c+dc) for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)] if in_bounds(r+dr, c+dc)]

        all_cells = [(r, c) for r in range(ch) for c in range(cw)]

        # Add first cell
        start = rng.choice(all_cells)
        in_maze.add(start)
        grid[2*start[0]+1, 2*start[1]+1] = PATH

        remaining = [c for c in all_cells if c not in in_maze]
        rng.shuffle(remaining)

        for cell in remaining:
            if cell in in_maze:
                continue
            # Random walk from cell until we hit maze
            path = [cell]
            path_set = {cell: 0}
            curr = cell
            while curr not in in_maze:
                nbs = nb(*curr)
                nxt = rng.choice(nbs)
                if nxt in path_set:
                    # Loop erasure
                    loop_start = path_set[nxt]
                    path = path[:loop_start+1]
                    path_set = {p: i for i, p in enumerate(path)}
                else:
                    path.append(nxt)
                    path_set[nxt] = len(path) - 1
                curr = nxt

            # Carve the path into the maze
            for i, (r, c) in enumerate(path):
                grid[2*r+1, 2*c+1] = PATH
                in_maze.add((r, c))
                if i > 0:
                    pr, pc = path[i-1]
                    grid[r+pr+1, c+pc+1] = PATH

        return grid

    # ----------------------------------------------------------
    @classmethod
    def hybrid(cls, h: int, w: int, seed: int = None) -> np.ndarray:
        """
        Hybrid: Generate a large backtracker base without making illegal shortcuts.
        Produces complex but strict topology.
        """
        rng = random.Random(seed)
        # Use straight-biased backtracker as the foundation
        grid = cls.backtracker(h, w, seed=rng.randint(0, 99999))
        return grid

    # ----------------------------------------------------------
    @classmethod
    def generate(cls, h: int, w: int, algorithm: str = 'backtracker',
                 seed: int = None) -> np.ndarray:
        """Dispatch to the correct algorithm."""
        # Ensure odd dimensions
        h = h if h % 2 == 1 else h + 1
        w = w if w % 2 == 1 else w + 1
        alg_map = {
            'backtracker': cls.backtracker,
            'prim':        cls.prim,
            'wilson':      cls.wilson,
            'hybrid':      cls.hybrid,
        }
        fn = alg_map.get(algorithm, cls.backtracker)
        grid = fn(h, w, seed=seed)
        return grid


# ============================================================
#  FOG OF WAR
# ============================================================
class FogOfWar:
    """
    Tracks agent visibility. Cells within vision_radius of the agent
    are 'visible'; discovered cells are 'explored' (remembered but not lit).
    """
    def __init__(self, maze_h: int, maze_w: int, vision_radius: int = 4):
        self.h = maze_h
        self.w = maze_w
        self.radius = vision_radius
        self.visible:   np.ndarray = np.zeros((maze_h, maze_w), dtype=bool)
        self.explored:  np.ndarray = np.zeros((maze_h, maze_w), dtype=bool)

    def update(self, agent_r: int, agent_c: int):
        self.visible[:] = False
        for dr in range(-self.radius, self.radius + 1):
            for dc in range(-self.radius, self.radius + 1):
                if dr*dr + dc*dc <= self.radius*self.radius:
                    r, c = agent_r + dr, agent_c + dc
                    if 0 <= r < self.h and 0 <= c < self.w:
                        self.visible[r, c] = True
                        self.explored[r, c] = True

    def coverage(self) -> float:
        return float(self.explored.sum()) / (self.h * self.w)

    def reset(self):
        self.visible[:] = False
        self.explored[:] = False


# ============================================================
#  DYNAMIC OBSTACLE (Homing Trap)
# ============================================================
class DynamicObstacle:
    """
    A moving trap that pursues the agent using a biased random walk.
    On hard levels, multiple traps increase hunting pressure.
    """
    def __init__(self, maze: np.ndarray, start_r: int, start_c: int):
        self.r = start_r
        self.c = start_c
        self.maze = maze

    def _free(self, r: int, c: int) -> bool:
        h, w = self.maze.shape
        return 0 <= r < h and 0 <= c < w and self.maze[r, c] == PATH

    def move_toward(self, agent_r: int, agent_c: int, aggression: float = 0.6):
        """
        Move with probability `aggression` toward agent, else random valid step.
        """
        dr = np.sign(agent_r - self.r)
        dc = np.sign(agent_c - self.c)
        candidates = [(self.r+dr, self.c), (self.r, self.c+dc),
                      (self.r-dr, self.c), (self.r, self.c-dc)]
        toward = [(self.r+dr, self.c+dc)]

        if random.random() < aggression and self._free(self.r+dr, self.c+dc):
            self.r += dr
            self.c += dc
        else:
            valid = [p for p in candidates if self._free(*p)]
            if valid:
                self.r, self.c = random.choice(valid)

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.r, self.c)


# ============================================================
#  A* PATHFINDER (Oracle comparison)
# ============================================================
def astar(maze: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """A* shortest path. Returns list of (r,c) tuples or empty if no path."""
    h, w = maze.shape

    def heur(r, c):
        return abs(r - goal[0]) + abs(c - goal[1])

    open_heap = [(heur(*start), 0, start, [start])]
    visited = set()

    while open_heap:
        f, g, curr, path = heapq.heappop(open_heap)
        if curr in visited:
            continue
        visited.add(curr)
        if curr == goal:
            return path
        r, c = curr
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and maze[nr,nc] == PATH and (nr,nc) not in visited:
                heapq.heappush(open_heap, (g+1+heur(nr,nc), g+1, (nr,nc), path+[(nr,nc)]))
    return []


# ============================================================
#  ENVIRONMENT ENGINE
# ============================================================
class MazeEnvironment:
    """
    Full RL environment wrapping the maze world.

    State encoding (flattened vector):
      • 9-cell local vision field (walls around agent)
      • Normalized agent position (r/H, c/W)
      • Normalized target position (tr/H, tc/W)
      • Normalized distance to target
      • Normalized distance to nearest trap
      • Fog coverage
      • Time pressure (steps / max_steps)
    Total state size: 9 + 2 + 2 + 1 + 1 + 1 + 1 = 17
    """
    STATE_SIZE  = 52
    ACTION_SIZE = 4  # up, down, left, right
    DELTAS      = [(-1,0),(1,0),(0,-1),(0,1)]

    def __init__(self, config: Dict = None):
        cfg = config or {}
        self.maze_h      = cfg.get('maze_h',     11)
        self.maze_w      = cfg.get('maze_w',     13)
        self.algorithm   = cfg.get('algorithm',  'backtracker')
        self.use_fog     = cfg.get('fog',         False)
        self.use_dynamic = cfg.get('dynamic',     False)
        self.use_portals = cfg.get('portals',     False)
        self.max_steps   = cfg.get('max_steps',   self.maze_h * self.maze_w * 4)

        self.maze:   Optional[np.ndarray] = None
        self.fog:    Optional[FogOfWar]   = None
        self.traps:  List[DynamicObstacle] = []
        self.portals: List[Tuple[Tuple,Tuple]] = []

        # State
        self.agent_r = 1
        self.agent_c = 1
        self.target_r = 1
        self.target_c = 1
        self.step_count = 0
        self.done = False
        self.last_action = -1

        # Metrics
        self.episode_reward    = 0.0
        self.cells_visited:    Set[Tuple] = set()
        self.visit_grid:       np.ndarray = np.zeros((1, 1))
        self.astar_optimal:    int        = 0
        self.total_episodes:   int        = 0
        self.success_count:    int        = 0
        self.seed:             int        = 0

        self.reset()

    # ----------------------------------------------------------
    def reset(self, config: Dict = None, seed: int = None):
        if config:
            self.maze_h      = config.get('maze_h',     self.maze_h)
            self.maze_w      = config.get('maze_w',     self.maze_w)
            self.algorithm   = config.get('algorithm',  self.algorithm)
            self.use_fog     = config.get('fog',         self.use_fog)
            self.use_dynamic = config.get('dynamic',     self.use_dynamic)
            self.use_portals = config.get('portals',     self.use_portals)
            self.max_steps   = self.maze_h * self.maze_w * 4

        self.seed = seed if seed is not None else random.randint(0, 2**31)
        self.maze = MazeGenerator.generate(self.maze_h, self.maze_w,
                                           self.algorithm, self.seed)
        H, W = self.maze.shape
        self.maze_h, self.maze_w = H, W

        # Find open cells
        open_cells = list(zip(*np.where(self.maze == PATH)))
        random.shuffle(open_cells)
        self.agent_r, self.agent_c = open_cells[0]
        self.target_r, self.target_c = open_cells[-1]

        # Ensure agent and target are not the same
        while (self.agent_r, self.agent_c) == (self.target_r, self.target_c):
            self.target_r, self.target_c = random.choice(open_cells)

        # Fog of war
        self.fog = FogOfWar(H, W, vision_radius=max(3, min(6, H//4)))
        if self.use_fog:
            self.fog.update(self.agent_r, self.agent_c)

        # Dynamic traps
        self.traps = []
        if self.use_dynamic:
            n_traps = min(3, len(open_cells) // 10)
            trap_starts = open_cells[len(open_cells)//3 : len(open_cells)//3 + n_traps]
            for tr, tc in trap_starts:
                if (tr, tc) != (self.agent_r, self.agent_c):
                    self.traps.append(DynamicObstacle(self.maze, tr, tc))

        # Portals
        self.portals = []
        if self.use_portals and len(open_cells) >= 10:
            n_portals = 2
            for i in range(n_portals):
                a = open_cells[i + 2]
                b = open_cells[-(i + 3)]
                self.portals.append((a, b))

        # A* oracle path length
        path = astar(self.maze, (self.agent_r, self.agent_c), (self.target_r, self.target_c))
        self.astar_optimal = len(path) - 1 if path else self.maze_h + self.maze_w

        # Reset counters
        self.step_count    = 0
        self.done          = False
        self.last_action   = -1
        self.episode_reward = 0.0
        self.cells_visited  = {(self.agent_r, self.agent_c)}
        self.visit_grid     = np.zeros((H, W), dtype=np.float32)
        self.visit_grid[self.agent_r, self.agent_c] = 1.0
        self.total_episodes += 1

        # return self._encode_state()
        return self._encode_state_discrete()

    # ----------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return self._encode_state(), 0.0, True, {}

        self.last_action = action
        dr, dc = self.DELTAS[action]
        nr, nc = self.agent_r + dr, self.agent_c + dc
        H, W = self.maze.shape

        # --- Reward shaping: potential-based ---
        old_dist = abs(self.agent_r - self.target_r) + abs(self.agent_c - self.target_c)

        # Move if valid
        moved = False
        if 0 <= nr < H and 0 <= nc < W and self.maze[nr, nc] == PATH:
            self.agent_r, self.agent_c = nr, nc
            moved = True

        new_dist = abs(self.agent_r - self.target_r) + abs(self.agent_c - self.target_c)
        self.step_count += 1

        # Visit tracking
        self.cells_visited.add((self.agent_r, self.agent_c))
        self.visit_grid[self.agent_r, self.agent_c] += 1.0

        # Portal teleport
        for (pa, pb) in self.portals:
            if (self.agent_r, self.agent_c) == pa:
                self.agent_r, self.agent_c = pb
            elif (self.agent_r, self.agent_c) == pb:
                self.agent_r, self.agent_c = pa

        # Dynamic trap movement & collision
        trap_hit = False
        for trap in self.traps:
            trap.move_toward(self.agent_r, self.agent_c, aggression=0.55)
            if trap.pos == (self.agent_r, self.agent_c):
                trap_hit = True

        # Fog update
        if self.use_fog:
            self.fog.update(self.agent_r, self.agent_c)

        # --- Compute reward ---
        #reward = self._compute_reward(moved, old_dist, new_dist, trap_hit)
        reward = self._compute_reward_discrete(trap_hit)
        self.episode_reward += reward

        # --- Check termination ---
        reached = (self.agent_r == self.target_r and self.agent_c == self.target_c)
        timeout = self.step_count >= self.max_steps
        self.done = reached or timeout or trap_hit

        if reached:
            self.success_count += 1

        info = {
            'reached':      reached,
            'timeout':      timeout,
            'trap_hit':     trap_hit,
            'steps':        self.step_count,
            'cells_visited': len(self.cells_visited),
            'fog_coverage':  self.fog.coverage() if self.use_fog else 1.0,
            'optimality':   self.astar_optimal / max(self.step_count, 1),
        }

        #return self._encode_state(), reward, self.done, info
        # OLD CONTINUOUS RETURN:
        # return self._encode_state(), reward, self.done, info
        # NEW DISCRETE RETURN:
        return self._encode_state_discrete(), reward, self.done, info


    # ----------------------------------------------------------
    # --- NEW: 0% CHEAT DISCRETE LOGIC ---
    def _encode_state_discrete(self) -> tuple:
        """Returns the exact, unaliased discrete coordinate."""
        return (self.agent_r, self.agent_c)

    def _compute_reward_discrete(self, trap_hit: bool) -> float:
        """Pure step penalties. No distance shaping, no double-curiosity."""
        r = -0.1  # Standard step penalty forces shortest path learning
        if trap_hit:
            r -= 5.0
            
        reached = (self.agent_r == self.target_r and self.agent_c == self.target_c)
        if reached:
            # Massive Jackpot to anchor the RL policy, scaling with path efficiency
            efficiency = max(0.0, self.astar_optimal / max(self.step_count, 1))
            r += 25.0 + 10.0 * efficiency

        if self.step_count >= self.max_steps and not reached:
            r -= 2.0  # Timeout penalty

        return float(np.clip(r, -20.0, 100.0))
      
    # ----------------------------------------------------------
    def _compute_reward(self, moved: bool, old_dist: int, new_dist: int,
                        trap_hit: bool) -> float:
        r = 0.0
        # --- MazE.py GENIUS MODE ALIGNMENT (0 Cheats) ---
        visit_count = self.visit_grid[self.agent_r, self.agent_c]
        
        # 1. Dynamic Penalty (Anti-Loitering, perfectly mimicking MazE.py)
        # -0.1 base living penalty, scales linearly with visits
        dynamic_penalty = -0.1 - (0.001 * visit_count)
        
        # 2. Curiosity Bonus (The "Fun" Factor)
        # kappa = 0.5 as used in MazE.py. We use the same square root decay.
        intrinsic_reward = (0.5 / np.sqrt(max(1.0, visit_count)))
        
        r += dynamic_penalty + intrinsic_reward

        if trap_hit:
            r -= 5.0                       # Caught by trap

        reached = (self.agent_r == self.target_r and self.agent_c == self.target_c)
        if reached:
            # Massive Jackpot to anchor the RL policy, scaling with path efficiency
            efficiency = max(0.0, self.astar_optimal / max(self.step_count, 1))
            r += 25.0 + 10.0 * efficiency

        if self.step_count >= self.max_steps and not reached:
            r -= 2.0                       # Timeout penalty

        return float(np.clip(r, -20.0, 40.0))

    # ----------------------------------------------------------
    def _encode_state(self) -> np.ndarray:
        """Encode the current world state as a normalized vector."""
        H, W = self.maze.shape
        r, c = self.agent_r, self.agent_c

        # 1. 25-cell High-Res local vision (5x5 centered on agent)
        vision = []
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    # Check if this cell is the objective
                    if (nr, nc) == (self.target_r, self.target_c):
                        cell = -1.0  # Distinct 'Goal' visual signal
                    else:
                        cell = float(self.maze[nr, nc])
                    
                    if self.use_fog and not self.fog.visible[nr, nc]:
                        cell = 0.5   # Unknown cell encoded as 0.5
                else:
                    cell = 1.0       # Out of bounds = wall
                vision.append(cell)

        # 2. 13-point Symmetrical Pheromone Diamond
        # We normalize this by log1p(x)/5.0 to keep values mostly in [0, 1] range.
        pheromones = []
        diamond_deltas = [
            (0,0), (-1,0), (1,0), (0,-1), (0,1), 
            (-1,-1), (-1,1), (1,-1), (1,1),
            (-2,0), (2,0), (0,-2), (0,2)
        ]
        for dr, dc in diamond_deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                v = self.visit_grid[nr, nc]
                norm_v = np.log1p(v) / 5.0
                pheromones.append(float(norm_v))
            else:
                pheromones.append(1.0) # Walls act as dead pheromone zones

        # 3. Global Telemetry (10-D)
        pos   = [r / H, c / W]
        tpos  = [self.target_r / H, self.target_c / W]
        
        # Directional vector to target
        dir_vec = [(self.target_r - r) / H, (self.target_c - c) / W]

        # 4. Normalized Manhattan distance
        dist = (abs(r - self.target_r) + abs(c - self.target_c)) / (H + W)

        # 5. Distance to nearest trap
        trap_dist = 1.0
        if self.traps:
            td = min(abs(tr.r - r) + abs(tr.c - c) for tr in self.traps)
            trap_dist = td / (H + W)

        # 6. Fog coverage (exploration completeness)
        fog_cov = self.fog.coverage() if self.use_fog else 1.0

        # 7. Time pressure
        time_pressure = self.step_count / max(self.max_steps, 1)

        # 8. Kinesthetic Momentum (4-D One-Hot)
        momentum = [1.0 if self.last_action == a else 0.0 for a in range(4)]

        # Total: 25(vision) + 13(pheromones) + 2(pos) + 2(tpos) + 2(dir) + 1(dist) + 1(trap) + 1(fog) + 1(time) + 4(mom) = 52
        state = vision + pheromones + pos + tpos + dir_vec + [dist, trap_dist, fog_cov, time_pressure] + momentum
        return np.array(state, dtype=np.float32)

    # ----------------------------------------------------------
    def render_ascii(self, width: int = None) -> Tuple[str, str]:
        """Returns (colored_grid, legend) for display."""
        H, W = self.maze.shape
        rows = []
        symbols = {
            'wall':    '██',
            'path':    '  ',
            'agent':   '🤖',
            'target':  '🏁',
            'trap':    '💀',
            'portal':  '🌀',
            'fog':     '░░',
            'explored':'  ',
        }

        trap_positions = {t.pos for t in self.traps}
        portal_positions = {pa for pa, pb in self.portals} | {pb for pa, pb in self.portals}

        for r in range(H):
            row = ''
            for c in range(W):
                is_agent  = (r == self.agent_r  and c == self.agent_c)
                is_target = (r == self.target_r and c == self.target_c)
                is_trap   = (r, c) in trap_positions
                is_portal = (r, c) in portal_positions
                is_fog    = self.use_fog and not self.fog.explored[r, c]
                is_visible = not self.use_fog or self.fog.visible[r, c]

                if is_agent:
                    row += symbols['agent']
                elif is_target and is_visible:
                    row += symbols['target']
                elif is_trap and is_visible:
                    row += symbols['trap']
                elif is_portal and is_visible:
                    row += symbols['portal']
                elif self.maze[r, c] == WALL:
                    if is_fog:
                        row += '▓▓'
                    else:
                        row += symbols['wall']
                else:
                    if is_fog:
                        row += symbols['fog']
                    elif self.visit_grid[r, c] > 0 and is_visible:
                        row += '·· '
                    else:
                        row += symbols['path']
            rows.append(row)

        grid_str = '\n'.join(rows)
        legend = f"Steps: {self.step_count}/{self.max_steps} | Visited: {len(self.cells_visited)} cells | Fog: {'ON' if self.use_fog else 'OFF'}"
        return grid_str, legend

    # ----------------------------------------------------------
    def get_render_data(self) -> Dict:
        """Returns structured data for Streamlit rendering."""
        H, W = self.maze.shape
        trap_positions = {t.pos for t in self.traps}
        portal_positions_a = {pa for pa, pb in self.portals}
        portal_positions_b = {pb for pa, pb in self.portals}

        data = {
            'maze':         self.maze.tolist(),
            'H': H, 'W': W,
            'agent':        (self.agent_r, self.agent_c),
            'target':       (self.target_r, self.target_c),
            'traps':        list(trap_positions),
            'portals_a':    list(portal_positions_a),
            'portals_b':    list(portal_positions_b),
            'fog_explored': self.fog.explored.tolist() if self.fog else None,
            'fog_visible':  self.fog.visible.tolist()  if self.fog else None,
            'visit_grid':   self.visit_grid.tolist(),
            'step_count':   self.step_count,
            'max_steps':    self.max_steps,
            'use_fog':      self.use_fog,
            'algorithm':    self.algorithm,
            'astar_optimal':self.astar_optimal,
        }
        return data

    # ----------------------------------------------------------
    def get_stats(self) -> Dict:
        return {
            'step_count':    self.step_count,
            'max_steps':     self.max_steps,
            'maze_size':     f"{self.maze_h}×{self.maze_w}",
            'algorithm':     self.algorithm,
            'cells_visited': len(self.cells_visited),
            'total_cells':   int((self.maze == PATH).sum()),
            'episode_reward':round(self.episode_reward, 2),
            'total_episodes':self.total_episodes,
            'success_count': self.success_count,
            'success_rate':  round(self.success_count / max(self.total_episodes, 1), 3),
            'astar_optimal': self.astar_optimal,
            'traps':         len(self.traps),
            'portals':       len(self.portals),
            'fog':           self.use_fog,
            'fog_coverage':  round(self.fog.coverage(), 3) if self.use_fog else 1.0,
        }

    # ----------------------------------------------------------
    def get_heatmap_data(self) -> np.ndarray:
        """Returns normalized visit frequency heatmap."""
        vg = self.visit_grid.copy()
        if vg.max() > 0:
            vg /= vg.max()
        return vg

    # ----------------------------------------------------------
    def get_astar_path(self) -> List[Tuple[int,int]]:
        return astar(self.maze, (self.agent_r, self.agent_c), (self.target_r, self.target_c))
