# 🐸 Q-Frogger: Simplified Frogger with Q-Learning

Q-Frogger is a minimalist recreation of the classic arcade game *Frogger*, where a frog attempts to cross a hazardous environment (e.g., roads and rivers) without dying. What makes this version special? The frog learns to master the game using **Q-learning**, a type of reinforcement learning algorithm.

## 🎯 Goal

Train a frog agent using Q-learning so that it can successfully cross the environment and win through experience and learning.

---

## 🧠 How It Works

### Game Environment
- The world is represented as a 2D grid.
- Rows may contain moving cars.
- The frog starts at the bottom and must reach the top.
- One step at a time, the frog can move: `up`, `down`, `left`, or `right`.

### Q-Learning Agent
- The agent learns an optimal policy using the Q-learning algorithm.
- State: 5x5 grid of hazards neawrby the frog with the frog at the center.
- Actions: Move in 4 directions.
- Rewards:
  - +50 for reaching the goal (winning).
  - -20 for dying (collision or falling in water).
  - -0.1 per move to encourage faster solutions.

### Features
- Simple text/grid-based.
- Training mode: Run many episodes to train the agent.
- Evaluation mode: Watch the trained frog in action.

---
