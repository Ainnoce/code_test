import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
TAU = 5e-3


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * 2
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        q = torch.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)  # [3,] -> [1, 3]
        next_state = np.expand_dims(next_state, axis=0)  # [3,] -> [1, 3]
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # state is a tuple of 1D arrays batch_size * [1, state_dim]
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )

        return (
            np.concatenate(states),  # [batch_size, state_dim]
            # [batch_size, action_dim]; [batch_size,] if action_dim=1
            np.array(actions),
            np.array(rewards),
            np.concatenate(next_states),  # [batch_size, state_dim]
            np.array(dones),
        )

    def len(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_capacity=MEMORY_CAPACITY):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(replay_buffer_capacity)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if self.replay_buffer.len() < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            BATCH_SIZE
        )

        states = torch.FloatTensor(states).to(DEVICE)
        # vstack to compatible convert list of arrays to 2D array
        actions = torch.FloatTensor(np.vstack(actions)).to(DEVICE)
        # unsqueeze to convert [batch_size,] to [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # update critic
        next_actions = self.actor_target(next_states)
        target_q = rewards + (1 - dones) * GAMMA * self.critic_target(
            next_states, next_actions
        )
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update_actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
