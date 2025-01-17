import multiprocessing
import random
import time

import numpy as np

from enviroment import JumperFrogEnv

TIMEOUT = 2  # Limit czasu w sekundach


def train_q_learning_wrapper(params):
    alpha, gamma, epsilon, epsilon_decay = params
    env = JumperFrogEnv()  # 🚀 Tworzymy nowe środowisko dla każdego procesu
    q_table, rewards = train_q_learning(env, 1000, alpha, gamma, epsilon, epsilon_decay)
    score = evaluate_agent(env, q_table)  # 🏆 Ewaluacja wyniku na podstawie Q-tablicy
    return params, score, q_table


def train_q_learning(env, episodes, alpha, gamma, epsilon, epsilon_decay, max_time=2.0):
    q_table = {}
    rewards = []
    start_time = time.time()  # ⏳ Start pomiaru czasu

    for episode in range(episodes):
        if time.time() - start_time > max_time:  # ⏳ Przerwij jeśli przekroczono czas
            print(f"Training stopped early after {episode} episodes (took too long)")
            return q_table, rewards

        state, _ = env.reset()
        state = tuple(state.flatten())
        total_reward = 0
        done = False

        while not done:
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
        state = tuple(state.flatten())  # Konwersja stanu na krotkę
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
    TIMEOUT = 2.0  # **Maksymalny czas trwania jednego treningu (sekundy)**

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:  # 🌍 Używamy wszystkich rdzeni CPU
        results = []

        for params in hyperparameters:
            print(f"Testing parameters: {params}")

            async_result = pool.apply_async(train_q_learning_wrapper, (params,))  # 🚀 Asynchroniczny start procesu

            try:
                # ⏳ Oczekujemy na wynik w czasie <= TIMEOUT
                result = async_result.get(timeout=TIMEOUT)
                results.append(result)  # ✅ Dodajemy tylko zakończone w czasie wyniki
            except multiprocessing.TimeoutError:
                print(f"Skipping parameters {params} (execution took too long)")
                continue  # ⏭️ Przechodzimy do następnej kombinacji

    # **Wybieramy najlepszy wynik**
    for params, score, q_table in results:
        print(f"Parameters: {params}, Score: {score}")

        if score > best_score:
            best_score = score
            best_params = params
            best_q_table = q_table

    print(f"Best Parameters: {best_params} with Score: {best_score}")

    return best_params, best_score, best_q_table

def visualize_steps(env: JumperFrogEnv, qtable: dict):
    # **Wizualizacja najlepszego agenta**
    env = JumperFrogEnv()
    for _ in range(5):
        print("Starting new simulation")
        state, _ = env.reset()
        done = False
        while not done:
            env.render()
            action = np.argmax([qtable.get((tuple(state.flatten()), a), 0) for a in range(env.action_space.n)])
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                print("Done because", "terminated" if terminated else "truncated")
        env.render()
        print('-'*50)


if __name__ == "__main__":
    params = (0.125, 0.775, 0.85, 0.995)
    alpha, gamma, epsilon, epsilon_decay = params
    env = JumperFrogEnv()
    qtable, _ = train_q_learning(env, 10000, alpha, gamma, epsilon, epsilon_decay, max_time=5.0)
    visualize_steps(env, qtable)
