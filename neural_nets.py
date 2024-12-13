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
        '''
        Neural network for learning a policy. Forward on this module takes in observation states and produces probability
        distributions over the action space.

        Args:
            input_dim: Dimensionality of the observation/state space vector that will be passed to the network.

            output_dim: Dimensionality of the action space.

            hidden_dim: Number of nodes in the two hidden layers.
        '''
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        '''
        pi(s)

        Given an observation/state tensor x, return corresponding probability distributions over the action space.
        '''
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        '''
        Neural network for learning a value function. Forward on this module takes in observation states and produces
        an estimate for the value function at each state.

        Args:
            input_dim: Dimensionality of the observation/state space vector that will be passed to the network.

            hidden_dim: Number of nodes in the two hidden layers.
        '''
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        '''
        V(s)

        Given an observation/state tensor x, return corresponding value functions for each state.
        '''
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def compute_advantages(rewards, dones, values, gamma=0.99, lam=0.95):
    '''
    Generalized advantage estimation (GAE) on a single trajectory. gamma is a discount factor
    against distant rewards (and introduces bias), and lam is a bias-variance tradeoff factor
    that controls this bias.
    '''
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (1 - dones[i]) * values[i + 1] - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages



def rollout_trajectory_details(n, env, steps_per_trajectory, policy_net, value_net, gamma, lam):
    '''
    Gets summary statistics for `n` trajectories.
    '''
    obs_buf, act_buf, logp_buf = [], [], []
    ep_rews = []
    traj_rewards = []
    all_advantages = []
    all_returns = []

    for _ in range(n):
        obs = env.reset()
        done = False
        traj_reward = 0
        count = 0
        for _ in range(steps_per_trajectory):
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
            traj_reward += reward

            obs = next_obs

            count += 1

            if done:
                break

        # Compute value and advantage targets
        traj_rewards.append(traj_reward)
        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32)
        values = value_net(obs_tensor).detach().numpy()
        values = np.append(values, 0)  # Bootstrap value for final state

        advantages = compute_advantages(ep_rews[-count:], dones=[0] * count, values=values[-count-1:], gamma=gamma, lam=lam)
        returns = advantages + values[-count-1:-1]

        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_advantages.append(advantages)
        all_returns.append(returns)

    advantages = torch.cat(all_advantages, 0)
    returns = np.concatenate(all_returns, 0)

    return (
        torch.tensor(np.array(obs_buf), dtype=torch.float32), 
        torch.tensor(np.array(act_buf), dtype=torch.int32), 
        torch.tensor(np.array(logp_buf), dtype=torch.float32), 
        advantages, 
        torch.tensor(np.array(returns), dtype=torch.float32),
        traj_rewards
    )


def ppo(
    env_name, policy_lr=3e-4, value_lr=1e-3, epochs=100, steps_per_trajectory=2048, 
    gamma=0.99, lam=0.95, clip_ratio=0.2, target_kl=0.01, train_policy_iters=80, 
    train_value_iters=80, trajectories_per_epoch=10,
):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = PolicyNetwork(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    for epoch in range(epochs):
        obs_tensor, act_tensor, logp_old, advantages, returns, rewards = rollout_trajectory_details(
            trajectories_per_epoch, env, steps_per_trajectory, policy_net, value_net, gamma, lam
        )

        print(
            "Epoch: ", epoch, 
            "\tAverage Reward: ", np.mean(rewards), 
            "\tStandard Deviation: ", np.std(rewards, ddof=1),
        )

        # PPO Policy Update
        for _ in range(train_policy_iters):
            policy_optimizer.zero_grad()

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
            loss_value = ((values - returns)**2).mean()
            
            loss_value.backward()
            value_optimizer.step()

    env.close()
    return policy_net, value_net


if __name__ == "__main__":
    main()
