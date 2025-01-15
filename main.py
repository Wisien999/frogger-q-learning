from enviroment import JumperFrogEnv

if __name__ == "__main__":
    env = JumperFrogEnv(render_mode="human")
    obs, info = env.reset()

    env.render()

    done = False
    total_reward = 0

    for step in range(50):  # limit steps in an episode
        action = env.action_space.sample()  # choose random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

        if terminated or truncated:
            print(f"Episode finished after {step+1} steps. Total reward: {total_reward}")
            break

    env.close()

