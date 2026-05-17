import gym
import numpy as np
import os
import pygame
import torch
import torch.nn as nn


from rl_models.ddpg.ddpg_agent import Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# environment
env = gym.make(id="Pendulum-v1", render_mode="rgb_array")
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# hyperparameters
NUM_EPISODES = 30
NUM_STEPS = 200

current_path = os.path.dirname(os.path.realpath(__file__))
models_path = os.path.join(current_path, "models")
model_file = os.path.join(models_path, "ddpg_actor.pth")
actor = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(DEVICE)
actor.load_state_dict(torch.load(model_file, map_location=DEVICE))
actor.eval()

# init pygame
pygame.init()
screen_size = (600, 400)
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()
pygame.display.set_caption("DDPG Test")


def process_frame(frame):
    frame = np.transpose(frame, (1, 0, 2))  # [400, 600, 3] -> [600, 400, 3]
    frame = pygame.surfarray.make_surface(frame)
    return pygame.transform.scale(frame, screen_size)


# run test episodes
for episode_i in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    for step_i in range(NUM_STEPS):
        action = actor(torch.FloatTensor(state).to(DEVICE)).cpu().detach().numpy()[0]
        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state
        episode_reward += reward

        # render
        frame = env.render()
        frame = process_frame(frame)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(30)

        if terminated:
            break
    print(f"Episode: {episode_i}, Reward: {episode_reward:.2f}")

pygame.quit()
env.close()
