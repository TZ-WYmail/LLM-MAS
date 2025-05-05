
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

from DisasterEnv import DisasterResponseEnv
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # 定义神经网络
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        action = torch.argmax(act_values, dim=1).item()
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]

        target_q_values = q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * max_next_q_values[i]

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    duration = 20
    env = DisasterResponseEnv(duration)
    state_size = env.observation_space.shape[0]
    action_size_rescue = env.action_space_rescue.n
    action_size_resource = env.action_space_resource.n
    action_size_rebuild = env.action_space_rebuild.n

    agent_rescue = DQNAgent(state_size, action_size_rescue)
    agent_resource = DQNAgent(state_size, action_size_resource)
    agent_rebuild = DQNAgent(state_size, action_size_rebuild)

    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward_rescue = 0
        total_reward_resource = 0
        total_reward_rebuild = 0
        while not done:
            action_rescue = agent_rescue.act(state)
            action_resource = agent_resource.act(state)
            action_rebuild = agent_rebuild.act(state)

            de_action_rescue = {'action': action_rescue, 'quantity': 1}
            de_action_resource = {'action': action_resource,
                                  'quantity': {'food': 1, 'water': 1, 'medical': 1, 'workers': 1}}
            de_action_rebuild = {'action': action_rebuild, 'quantity': 1}

            next_state, reward_rescue, reward_resource, reward_rebuild, done, _ = env.step(de_action_rebuild, de_action_rescue, de_action_resource)

            agent_rescue.remember(state, action_rescue, reward_rescue, next_state, done)
            agent_rescue.replay()

            agent_resource.remember(state, action_resource, reward_resource, next_state, done)
            agent_resource.replay()

            agent_rebuild.remember(state, action_rebuild, reward_rebuild, next_state, done)
            agent_rebuild.replay()

            state = next_state
            total_reward_rescue += reward_rescue
            total_reward_resource += reward_resource
            total_reward_rebuild += reward_rebuild

        print(
            f"Episode {episode + 1}: Rescue Reward = {total_reward_rescue}, Resource Reward = {total_reward_resource}, Rebuild Reward = {total_reward_rebuild}")
