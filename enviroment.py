import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class JumperFrogEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, width=7, height=5, num_cars_per_lane=2, view_size=5):
        super(JumperFrogEnv, self).__init__()
        self.width = width
        self.height = height
        self.num_cars_per_lane = num_cars_per_lane
        self.view_size = view_size  # Size of the observation grid (view_size x view_size)
        self.action_space = spaces.Discrete(4)  # Actions: up, down, left, right
        self.observation_space = spaces.Box(
            low=-self.width,  # Velocity of cars or 0
            high=self.width,  # Max velocity in any direction
            shape=(view_size, view_size),  # Observation grid
            dtype=np.float32
        )
        self.render_mode = render_mode
        self.frog_x = None
        self.frog_y = None
        self.cars = []

    def _initialize_cars(self):
        self.cars = []
        for lane_idx in range(1, self.height):  # Skip the top safe row
            direction = 1 if lane_idx % 2 == 0 else -1
            for _ in range(self.num_cars_per_lane):
                car_x = random.randint(0, self.width - 1)
                self.cars.append([car_x, lane_idx, direction])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frog_x = self.width // 2
        self.frog_y = self.height - 1
        self._initialize_cars()
        return self._get_obs(), {}

    def step(self, action):
        if action == 0:  # up
            self.frog_y = max(0, self.frog_y - 1)
        elif action == 1:  # down
            self.frog_y = min(self.height - 1, self.frog_y + 1)
        elif action == 2:  # left
            self.frog_x = max(0, self.frog_x - 1)
        elif action == 3:  # right
            self.frog_x = min(self.width - 1, self.frog_x + 1)

        for car in self.cars:
            car[0] += car[2]
            if car[0] < 0:
                car[0] = self.width - 1
            elif car[0] >= self.width:
                car[0] = 0

        reward = -0.1
        terminated = False
        for car_x, car_y, _ in self.cars:
            if car_x == self.frog_x and car_y == self.frog_y:
                reward = -20.0
                terminated = True
                break

        if self.frog_y == 0 and not terminated:
            reward = 50.0
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """
        Returns a 2D grid centered on the frog.
        The view is X x X (view_size x view_size) and shows car velocities or 0 if no car is present.
        The frog is always at the center row.
        """
        half_view = self.view_size // 2
        grid = np.zeros((self.view_size, self.view_size), dtype=np.float32)

        for car_x, car_y, direction in self.cars:
            dx = car_x - self.frog_x
            dy = car_y - self.frog_y

            if -half_view <= dx <= half_view and -half_view <= dy <= half_view:
                grid[dy + half_view, dx + half_view] = direction

        return grid

    def render(self):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for (cx, cy, _) in self.cars:
            grid[cy][cx] = "C"
        grid[self.frog_y][self.frog_x] = "F"
        print("=" * (self.width + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("=" * (self.width + 2))

    def close(self):
        pass
