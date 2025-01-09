import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class JumperFrogEnv(gym.Env):
    """
    A simplified Frogger-like environment.
    The frog starts at the bottom row (row = max_row - 1) and tries
    to reach the top row (row = 0) without getting hit by cars.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, width=7, height=5, num_cars_per_lane=2):
        super(JumperFrogEnv, self).__init__()

        # Grid size
        self.width = width
        self.height = height

        # Number of cars in each lane
        self.num_cars_per_lane = num_cars_per_lane

        # Define action and observation space
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation: we can store it as a 2D or flattened array,
        # but for simplicity, let's store (frog_x, frog_y, car_positions).
        # We'll define a simplified Box observation. The shape can vary.
        # For demonstration, let's do:
        #   frog_x, frog_y in [0, width-1] and [0, height-1]
        #   cars each has (x, lane_index) or we store them in a single array
        #
        # However, a truly complete representation might be more complex.
        # Here, we'll keep it simple and store positions in a 1D array
        # with length = number_of_cars * 2 plus frog x,y.
        max_cars = self.num_cars_per_lane * (self.height - 1)  # no cars in the top "safe" row
        obs_len = 2 + 2 * max_cars  # frog_x, frog_y, + each car (x,y)
        low = np.zeros(obs_len, dtype=np.float32)
        high = np.array([max(self.width, self.height)] * obs_len, dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Render mode
        self.render_mode = render_mode

        # Internal variables
        self.frog_x = None
        self.frog_y = None
        # We'll keep cars in a list of tuples (x, y, direction),
        # where direction = +1 (moves right) or -1 (moves left)
        self.cars = []

    def _initialize_cars(self):
        """
        Randomly place cars in each lane (row). 
        Each lane except the top (row 0) can have cars.
        Lane row i is 'height - i - 1'.
        """
        self.cars = []
        for lane_idx in range(1, self.height):  # skip row 0 (safe zone)
            row = lane_idx
            direction = 1 if lane_idx % 2 == 0 else -1  # alternate directions
            # Spawn a few cars in each lane
            for _ in range(self.num_cars_per_lane):
                car_x = random.randint(0, self.width - 1)
                # Store (x, y=row, direction)
                self.cars.append([car_x, row, direction])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the frog at the bottom row
        self.frog_x = self.width // 2
        self.frog_y = self.height - 1

        # Initialize cars
        self._initialize_cars()

        # Return initial observation and info
        return self._get_obs(), {}

    def step(self, action):
        # 1. Apply the action
        if action == 0:   # up
            self.frog_y = max(0, self.frog_y - 1)
        elif action == 1: # down
            self.frog_y = min(self.height - 1, self.frog_y + 1)
        elif action == 2: # left
            self.frog_x = max(0, self.frog_x - 1)
        elif action == 3: # right
            self.frog_x = min(self.width - 1, self.frog_x + 1)
        else:
            pass  # no-op if you add it

        # 2. Move cars
        for car in self.cars:
            car[0] += car[2]
            # Wrap around horizontally if desired (like cars cycling on the road)
            if car[0] < 0:
                car[0] = self.width - 1
            elif car[0] >= self.width:
                car[0] = 0

        # 3. Check if frog is hit
        reward = -0.1  # small negative step cost
        terminated = False
        truncated = False

        for car_x, car_y, _ in self.cars:
            if car_x == self.frog_x and car_y == self.frog_y:
                reward = -10.0
                terminated = True
                break

        # 4. Check if frog reached the top
        if self.frog_y == 0 and not terminated:
            reward = 10.0
            terminated = True

        # 5. Construct new observation
        obs = self._get_obs()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """
        Flatten frog_x, frog_y, plus all car positions (x, y).
        """
        max_cars = self.num_cars_per_lane * (self.height - 1)
        # Create an array with length = 2 + 2 * max_cars
        obs = np.zeros(2 + 2 * max_cars, dtype=np.float32)
        obs[0] = self.frog_x
        obs[1] = self.frog_y

        idx = 2
        for i, car in enumerate(self.cars):
            if i >= max_cars:
                break
            obs[idx] = car[0]  # car x
            obs[idx + 1] = car[1]  # car y
            idx += 2
        return obs

    def render(self):
        """
        Render the grid to console (ASCII).
        Top row is y=0, bottom row is y=height-1.
        """
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        # Mark cars
        for (cx, cy, _) in self.cars:
            grid[cy][cx] = "C"
        # Mark frog
        grid[self.frog_y][self.frog_x] = "F"

        # Print row by row
        print("=" * (self.width + 2))
        for row_idx in range(self.height):
            print("|" + "".join(grid[row_idx]) + "|")
        print("=" * (self.width + 2))

    def close(self):
        pass

