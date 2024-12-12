import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from tqdm import tqdm



def main():
    ppo("CartPole-v1")




class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

def compute_advantages(rewards, dones, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (1 - dones[i]) * values[i + 1] - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

def ppo(env_name, policy_lr=3e-4, value_lr=1e-3, epochs=100, steps_per_epoch=2048, 
        gamma=0.99, lam=0.95, clip_ratio=0.2, target_kl=0.01, train_policy_iters=80, train_value_iters=80):
    
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = PolicyNetwork(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    for epoch in range(epochs):
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = [], [], [], [], []

        obs = env.reset()
        done = False
        ep_rews = []

        for _ in range(steps_per_epoch):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            probs = policy_net(obs_tensor)
            dist = Categorical(probs)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action)).item()

            next_obs, reward, done, _ = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            ep_rews.append(reward)

            obs = next_obs

            if done:
                break

            # if done:
            #     obs = env.reset()
            #     done = False
            #     ep_rews = []

        # Compute value and advantage targets
        print(f"Total Reward for Epoch {epoch+1}: ", np.sum(ep_rews))
        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32)
        values = value_net(obs_tensor).detach().numpy()
        values = np.append(values, 0)  # Bootstrap value for final state

        advantages = compute_advantages(ep_rews, dones=[0]*len(ep_rews), values=values, gamma=gamma, lam=lam)
        returns = advantages + values[:-1]

        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Policy Update
        for _ in range(train_policy_iters):
            policy_optimizer.zero_grad()
            
            obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32)
            act_tensor = torch.tensor(np.array(act_buf), dtype=torch.int32)
            logp_old = torch.tensor(np.array(logp_buf), dtype=torch.float32)

            probs = policy_net(obs_tensor)
            dist = Categorical(probs)
            logp = dist.log_prob(act_tensor)

            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            loss_policy = -(torch.min(ratio * advantages, clip_adv)).mean()

            kl = (logp_old - logp).mean().item()
            if kl > target_kl:
                break

            loss_policy.backward()
            policy_optimizer.step()

        # PPO Value Update
        for _ in range(train_value_iters):
            value_optimizer.zero_grad()
            
            values = value_net(obs_tensor).squeeze()
            loss_value = ((values - torch.tensor(returns, dtype=torch.float32))**2).mean()
            
            loss_value.backward()
            value_optimizer.step()

    env.close()
    return policy_net, value_net


if __name__ == "__main__":
    main()
