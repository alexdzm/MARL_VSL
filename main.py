import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from collections import deque
import random

# Initialize wandb
wandb.init(project="vsl-marl", config={
    "num_agents": 12,
    "num_episodes": 10000,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "buffer_size": 100000,
    "update_every": 4,
    "hidden_size": 128
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class VSLEnvironment:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.state_size = 6
        self.action_size = 5
        self.max_speed = 80
        self.min_speed = 20
        self.max_time_steps = 288
        self.rush_hour_morning = (7, 10)
        self.rush_hour_evening = (16, 19)
        
    def reset(self):
        self.current_time_step = 0
        self.speeds = np.ones(self.num_agents) * 60
        self.occupancies = np.ones(self.num_agents) * 0.2
        return self._get_states()
    
    def _get_states(self):
        states = []
        time_of_day = self.current_time_step / self.max_time_steps
        for i in range(self.num_agents):
            prev_action = 60 if i == 0 else self.speeds[i-1]
            next_speed = self.speeds[i+1] if i < self.num_agents - 1 else self.speeds[i]
            next_occupancy = self.occupancies[i+1] if i < self.num_agents - 1 else self.occupancies[i]
            state = [prev_action/self.max_speed, self.speeds[i]/self.max_speed, self.occupancies[i],
                     next_speed/self.max_speed, next_occupancy, time_of_day]
            states.append(np.array(state))
        return states
    
    def _is_rush_hour(self):
        hour = (self.current_time_step * 24) // self.max_time_steps
        return (self.rush_hour_morning[0] <= hour < self.rush_hour_morning[1]) or \
               (self.rush_hour_evening[0] <= hour < self.rush_hour_evening[1])
    
    def step(self, actions):
        # Update speeds based on VSL and improved car-following model
        for i in range(self.num_agents):
            vsl = [30, 40, 50, 60, 70][actions[i]]
            desired_speed = min(vsl, self.speeds[i] + np.random.normal(0, 3))
            if i > 0:
                safe_distance = max(2, self.speeds[i] * 0.036)  # 2 seconds rule
                actual_distance = 1 / (self.occupancies[i-1] + 1e-6)
                desired_speed = min(desired_speed, self.speeds[i-1] * (actual_distance / safe_distance))
            self.speeds[i] = np.clip(desired_speed, self.min_speed, self.max_speed)
        
        # Update occupancies with improved model
        rush_hour_factor = 1.5 if self._is_rush_hour() else 1.0
        for i in range(self.num_agents):
            if i > 0:
                inflow = min(self.speeds[i-1], self.speeds[i]) * self.occupancies[i-1]
            else:
                inflow = 60 * 0.2 * rush_hour_factor
            outflow = self.speeds[i] * self.occupancies[i]
            self.occupancies[i] += (inflow - outflow) * 0.01
            self.occupancies[i] = np.clip(self.occupancies[i], 0, 1)
        
        self.current_time_step += 1
        next_states = self._get_states()
        
        # Calculate rewards with improved reward function
        rewards = []
        for i in range(self.num_agents):
            vsl = [30, 40, 50, 60, 70][actions[i]]
            adaptability = -abs(self.speeds[i] - vsl) / self.max_speed
            safety = -abs(self.speeds[i] - (self.speeds[i-1] if i > 0 else self.speeds[i])) / self.max_speed
            mobility = self.speeds[i] / self.max_speed
            efficiency = 1 - self.occupancies[i]
            reward = (adaptability * 0.2 + safety * 0.3 + mobility * 0.3 + efficiency * 0.2)
            rewards.append(reward)
        
        done = self._check_termination()
        
        return next_states, rewards, done
    
    def _check_termination(self):
        if self.current_time_step >= self.max_time_steps:
            return True
        if all(o > 0.3 for o in self.occupancies) and self.current_time_step % 6 == 0:
            return True
        if all(s > 50 for s in self.speeds) and self.current_time_step % 12 == 0:
            return True
        return False

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (torch.FloatTensor(states).to(device),
                torch.LongTensor(actions).to(device),
                torch.FloatTensor(rewards).to(device),
                torch.FloatTensor(next_states).to(device),
                torch.FloatTensor(dones).to(device))
    
    def __len__(self):
        return len(self.memory)

def train_marl(num_agents, num_episodes):
    env = VSLEnvironment(num_agents)
    actors = [Actor(env.state_size, env.action_size, wandb.config.hidden_size).to(device) for _ in range(num_agents)]
    critics = [Critic(env.state_size, env.action_size, wandb.config.hidden_size).to(device) for _ in range(num_agents)]
    actor_targets = [Actor(env.state_size, env.action_size, wandb.config.hidden_size).to(device) for _ in range(num_agents)]
    critic_targets = [Critic(env.state_size, env.action_size, wandb.config.hidden_size).to(device) for _ in range(num_agents)]
    
    for actor, actor_target in zip(actors, actor_targets):
        actor_target.load_state_dict(actor.state_dict())
    for critic, critic_target in zip(critics, critic_targets):
        critic_target.load_state_dict(critic.state_dict())
    
    actor_optimizers = [optim.Adam(actor.parameters(), lr=wandb.config.learning_rate) for actor in actors]
    critic_optimizers = [optim.Adam(critic.parameters(), lr=wandb.config.learning_rate) for critic in critics]
    
    replay_buffer = ReplayBuffer(wandb.config.buffer_size, wandb.config.batch_size)
    
    for episode in range(num_episodes):
        states = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            actions = []
            for i, actor in enumerate(actors):
                state = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
                action_probs = actor(state).squeeze(0).detach().cpu().numpy()
                action = np.random.choice(env.action_size, p=action_probs)
                actions.append(action)
            
            next_states, rewards, done = env.step(actions)
            episode_reward += sum(rewards)
            
            for i in range(num_agents):
                replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], done)
            
            if len(replay_buffer) > wandb.config.batch_size and step % wandb.config.update_every == 0:
                for i in range(num_agents):
                    experiences = replay_buffer.sample()
                    states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = experiences
                    
                    # Update Critic
                    next_actions = actor_targets[i](next_states_batch).detach()
                    Q_targets_next = critic_targets[i](next_states_batch, next_actions).detach().squeeze(-1)
                    Q_targets = rewards_batch + (wandb.config.gamma * Q_targets_next * (1 - dones_batch))
                    Q_expected = critics[i](states_batch, F.one_hot(actions_batch, env.action_size).float()).squeeze(-1)
                    critic_loss = F.mse_loss(Q_expected, Q_targets)
                    
                    critic_optimizers[i].zero_grad()
                    critic_loss.backward()
                    critic_optimizers[i].step()
                    
                    # Update Actor
                    actor_loss = -critics[i](states_batch, actors[i](states_batch)).mean()
                    
                    actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    actor_optimizers[i].step()
                    
                    # Update target networks
                    for param, target_param in zip(actors[i].parameters(), actor_targets[i].parameters()):
                        target_param.data.copy_(wandb.config.tau * param.data + (1.0 - wandb.config.tau) * target_param.data)
                    for param, target_param in zip(critics[i].parameters(), critic_targets[i].parameters()):
                        target_param.data.copy_(wandb.config.tau * param.data + (1.0 - wandb.config.tau) * target_param.data)
            
            states = next_states
            step += 1
            
            wandb.log({
                "step_reward": sum(rewards),
                "step": step + episode * env.max_time_steps
            })
        
        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward,
            "episode_length": env.current_time_step,
            "avg_reward_per_step": episode_reward / env.current_time_step
        })
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {episode_reward/env.current_time_step:.4f}, Duration: {env.current_time_step}")

# Run the training
num_agents = wandb.config.num_agents
num_episodes = wandb.config.num_episodes
train_marl(num_agents, num_episodes)

wandb.finish()