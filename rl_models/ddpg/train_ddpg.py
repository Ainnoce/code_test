import gym
import numpy as np
import os
import time
import torch

from rl_models.ddpg.ddpg_agent import DDPGAgent

# initialize environment
env = gym.make(id="Pendulum-v1")
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# hyperparameters
NUM_EPISODES = 100
NUM_STEPS = 200
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
ACTION_BOUND = [-2.0, 2.0]


agent = DDPGAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

REWARD_BUFFER = np.empty(shape=NUM_EPISODES)

for episode_i in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    for step_i in range(NUM_STEPS):
        epsilon = np.interp(
            x=episode_i * NUM_STEPS + step_i,
            xp=[0, EPSILON_DECAY],
            fp=[EPSILON_START, EPSILON_END],
        )
        random_sample = np.random.random()
        if random_sample < epsilon:
            action = np.random.uniform(
                low=ACTION_BOUND[0], high=ACTION_BOUND[1], size=ACTION_DIM
            )
        else:
            action = agent.get_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        agent.replay_buffer.push(state, action, reward, next_state, terminated)

        state = next_state
        episode_reward += reward

        agent.update()
        if terminated:
            break
    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode: {episode_i}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

# save model
current_path = os.path.dirname(os.path.realpath(__file__))
models_path = os.path.join(current_path, "models")
current_time = time.strftime("%Y%m%d_%H%M%S")

actor_model_path = os.path.join(models_path, f"ddpg_actor_{current_time}.pth")
critic_model_path = os.path.join(models_path, f"ddpg_critic_{current_time}.pth")
torch.save(agent.actor.state_dict(), actor_model_path)
torch.save(agent.critic.state_dict(), critic_model_path)

env.close()
