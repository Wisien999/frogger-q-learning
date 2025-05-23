import multiprocessing
from argparse import ArgumentParser
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle

from enviroment import JumperFrogEnv

TIMEOUT = 2.0  # Limit czasu w sekundach


def train_q_learning_wrapper(params):
    alpha, gamma, epsilon, epsilon_decay = params
    env = JumperFrogEnv()
    q_table, rewards = train_q_learning(env, 1000, alpha, gamma, epsilon, epsilon_decay)
    score = evaluate_agent(env, q_table)
    return params, score, q_table


def train_q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, max_time=2.0, epoch_max_time=1.0):
    q_table = {}
    rewards = []
    start_time = time.time()

    for episode in range(episodes):
        # if time.time() - start_time > max_time:
        #     print(f"Training stopped early after {episode} episodes (took too long)")
        #     return q_table, rewards

        state, _ = env.reset()
        state = tuple(state.flatten())
        total_reward = 0
        done = False
        simulation_start_time = time.time()
        skipped = False

        while not done:
            if time.time() - simulation_start_time > epoch_max_time:
                print("Long simulation. Skipping...")
                break 
            if time.time() - start_time > max_time:
                print(f"Training stopped early after {episode} episodes (took too long)")
                return q_table, rewards

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax([q_table.get((state, a), 0) for a in range(env.action_space.n)])

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_state.flatten())
            total_reward += reward

            if not terminated and not truncated:
                max_next_q = max([q_table.get((next_state, a), 0) for a in range(env.action_space.n)])
                q_table[(state, action)] = q_table.get((state, action), 0) + alpha * (
                        reward + gamma * max_next_q - q_table.get((state, action), 0)
                )
            elif terminated:
                q_table[(state, action)] = reward

            state = next_state
            done = terminated or truncated

        epsilon *= epsilon_decay
        rewards.append(total_reward)

    return q_table, rewards


def evaluate_agent(env, q_table, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        state = tuple(state.flatten())
        total_reward = 0
        done = False

        while not done:
            action = np.argmax([q_table.get((state, a), 0) for a in range(env.action_space.n)])
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_state.flatten())
            total_reward += reward
            state = next_state
            done = terminated or truncated

        total_rewards.append(total_reward)

    return np.mean(total_rewards)


def find_hyperparameters():
    # **Lista hiperparametrów do testowania**
    hyperparameters = [
        (alpha, gamma, epsilon, epsilon_decay)
        for alpha in [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.5]
        for gamma in [0.7, 0.75, 0.775, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
        for epsilon in [1.0, 0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.85]
        for epsilon_decay in [0.995, 0.99, 0.98]
    ]

    best_score = float('-inf')
    best_params = None
    best_q_table = None

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []

        for params in hyperparameters:
            print(f"Testing parameters: {params}")

            async_result = pool.apply_async(train_q_learning_wrapper, (params,))

            try:
                result = async_result.get(timeout=TIMEOUT)
                results.append(result)
            except multiprocessing.TimeoutError:
                print(f"Skipping parameters {params} (execution took too long)")
                continue

    for params, score, q_table in results:
        print(f"Parameters: {params}, Score: {score}")

        if score > best_score:
            best_score = score
            best_params = params
            best_q_table = q_table

    print(f"Best Parameters: {best_params} with Score: {best_score}")

    return best_params, best_score, best_q_table

def visualize_steps(env: JumperFrogEnv, qtable: dict):
    env = JumperFrogEnv()
    reward = 0
    print("Starting new simulation")
    state, _ = env.reset()
    done = False
    i = 0
    while not done and i < 20:
        # if i == 2:
        #     from pprint import pprint
        #     pprint(env._get_obs())
        env.render()
        action = np.argmax([qtable.get((tuple(state.flatten()), a), 0) for a in range(env.action_space.n)])
        state, rew, terminated, truncated, _ = env.step(action)
        reward += rew
        done = terminated or truncated
        i += 1
        if done or i == 19:
            print("Done because", "terminated" if terminated else "i reached")

    env.render()
    print("Complete (accumulated) reward over simulation", reward)
    print('-'*50)



def save_average_reward_plot(rewards, filename='average_reward_plot.png', window=50):
    """
    Saves a plot of the moving average reward of a Q-learning agent.
    
    :param rewards: List of rewards per episode.
    :param filename: File name to save the plot.
    :param window: Window size for moving average.
    """
    if not rewards:
        print("Error: Rewards list is empty.")
        return
    
    # Compute moving average
    avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards, label=f'Moving Average (window={window})', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Q-Learning Agent Average Reward Over Episodes')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(filename)
    plt.close()
    print(f'Plot saved as {filename}')




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    args = vars(parser.parse_args())

    env = JumperFrogEnv()

    if args['train']:
        params = (0.125, 0.775, 0.85, 0.995)
        alpha, gamma, epsilon, epsilon_decay = params

        print("Starting training...")
        qtable, rewards = train_q_learning(env, 500_000, alpha, gamma, epsilon, epsilon_decay, max_time=120.0)
        print("Training finished!")

        with open("q_table.pkl", "wb") as f:
            pickle.dump(qtable, f)

        print(f"Rewards list length:", len(rewards))

        print("Saving plots...")
        save_average_reward_plot(rewards, 'avg_reward_50.png', window=50)
        save_average_reward_plot(rewards, 'avg_reward_100.png', window=100)
        save_average_reward_plot(rewards, 'avg_reward_500.png', window=500)

        with open("q_table.pkl", "wb") as f:
            pickle.dump(qtable, f)
    else:
        with open("q_table.pkl", "rb") as f:
            qtable = pickle.load(f)

    print("Visualization of learned frog behaviour")
    visualize_steps(env, qtable)

