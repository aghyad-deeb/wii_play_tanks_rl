import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

import tanks  # Import the environment from tanks.py
from tanks import MOVE_TOP, MOVE_RIGHT, MOVE_BOTTOM, MOVE_LEFT, MOVE_TOP_RIGHT, MOVE_BOTTOM_RIGHT, MOVE_TOP_LEFT, MOVE_BOTTOM_LEFT

# Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
BATCH_SIZE = 128
PPO_EPOCHS = 10
NUM_EPISODES = 1000
TIMESTEPS_PER_EPISODE = 500
HIDDEN_SIZE = 256

class ActorCritic(nn.Module):
    def __init__(self, num_actions, input_size):
        super(ActorCritic, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.feature_extractor(x.flatten(1))
        return self.actor(x), self.critic(x)

    def act(self, x):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item(), probs.log_prob(action), value

    def evaluate(self, x, actions):
        logits, value = self.forward(x)
        probs = Categorical(logits=logits)
        log_probs = probs.log_prob(actions.squeeze(1))
        entropy = probs.entropy()
        return log_probs, value, entropy

def process_obs(obs):
    """
    Process the observation to be suitable for the neural network
    """
    agent_loc = obs["agent"].astype(float)
    targets_loc = np.array([target.astype(float) for target in obs["targets"]])
    bullets_loc = np.array([bullet[0].astype(float) for bullet in obs["bullets"]])
    walls_loc = np.array([wall.astype(float) for wall in obs["walls"]])
    map_ = np.array(obs["map"]).astype(float).flatten()
    return np.concatenate(
        (
            agent_loc, 
            targets_loc.flatten(), 
            bullets_loc.flatten(), 
            walls_loc.flatten(),
            map_
        )
    )

def train_ppo():
    env = tanks.main.env
    #print(f"{env.observation_space.spaces=}")
    num_movement_actions = env.action_space.nvec[0]
    num_shoot_actions = env.action_space.nvec[1]
    num_actions = num_movement_actions * num_shoot_actions
    print(f"num_movement_actions={num_movement_actions}\nnum_shoot_actions={num_shoot_actions}\n{num_actions=}")
    #observation_shape = env.observation_space.spaces.values()
    
    # Get input_size for the network by getting the shape of the observation
    state, _ = env.reset()
    processed_state = process_obs(state)
    input_size = processed_state.shape[0]

    model = ActorCritic(num_actions, input_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        total_reward = 0
        
        for t in range(TIMESTEPS_PER_EPISODE):
            processed_state = process_obs(state)
            
            action_movement, log_prob_movement, value_movement = model.act(processed_state)
            action_movement = (action_movement % num_movement_actions)
            
            action_shoot, log_prob_shoot, value_shoot = model.act(processed_state)
            action_shoot = (action_shoot % num_shoot_actions)

            next_state, reward, done, _, _ = env.step((action_movement, action_shoot))
            
            
            states.append(processed_state)
            actions.append(np.array((action_movement * num_shoot_actions) + action_shoot))
            log_probs.append(log_prob_movement + log_prob_shoot)
            rewards.append(reward)
            values.append(value_movement.detach().numpy())
            dones.append(done)

            total_reward += reward
            state = next_state
            
            if done:
                break

        # Convert lists to tensors for PPO update
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
        log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float)
        values = torch.tensor(np.array(values), dtype=torch.float).squeeze(1)
        dones = torch.tensor(dones, dtype=torch.float)
        
        advantages, returns = _compute_advantages_and_returns(
            rewards, values, dones, GAMMA, LAMBDA
        )

        for _ in range(PPO_EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                batch_states = states[i:i + BATCH_SIZE]
                batch_actions = actions[i:i + BATCH_SIZE]
                batch_log_probs = log_probs[i:i + BATCH_SIZE]
                batch_advantages = advantages[i:i + BATCH_SIZE]
                batch_returns = returns[i:i + BATCH_SIZE]

                batch_log_probs, batch_values, batch_entropy = model.evaluate(
                    batch_states, batch_actions
                )
                
                ratio = torch.exp(batch_log_probs - batch_log_probs.detach())
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - batch_values).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * batch_entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


def _compute_advantages_and_returns(rewards, values, dones, gamma, lambda_):
    """Compute advantages and returns for the PPO algorithm."""
    
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    advantage = 0
    return_ = 0

    # Iterate through time steps in reverse to calculate advantages and returns
    for t in reversed(range(len(rewards))):
        # Check if game has terminated
        non_terminal = 1 - dones[t]
        
        # Calculate TD target
        td_target = rewards[t] + gamma * non_terminal * (values[t+1] if t+1 < len(rewards) else 0)
        
        # Calculate temporal difference
        delta = td_target - values[t]
        
        # Calculate GAE
        advantage = delta + gamma * lambda_ * non_terminal * advantage
        advantages[t] = advantage
        
        # Calculate returns
        return_ = rewards[t] + gamma * non_terminal * return_
        returns[t] = return_

    return advantages, returns
if __name__ == "__main__":
    train_ppo()