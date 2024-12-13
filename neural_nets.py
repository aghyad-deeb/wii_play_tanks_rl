import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from tqdm import tqdm
from tanks import WiiTanks
from settings import FIELDHEIGHT, FIELDWIDTH, TRACKED_BULLETS, TRACKED_ENEMIES


def main():
    # cart_env = gym.make("CartPole-v1")
    # ppo(cart_env)
    tanks_env = WiiTanks()
    ppo(tanks_env)



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
        if len(x.shape) == 1:
            return torch.softmax(x[:9], dim=-1), torch.softmax(x[9:], dim=-1)
        else:
            return torch.softmax(x[:, :9], dim=-1), torch.softmax(x[:, 9:], dim=-1)

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
    obs_buf = []
    act_move_buf = []
    act_shoot_buf = []
    logp_move_buf = []
    logp_shoot_buf = []
    ep_rews = []
    traj_rewards = []
    all_advantages = []
    all_returns = []

    for _ in tqdm(range(n)):
        obs = env.reset()[0]
        done = False
        traj_reward = 0
        count = 0
        for _ in range(steps_per_trajectory):
            # Process observation before running network
            # Vectorize in order of bullets, enemies, player, walls

            # Position bullets randomly within the bullets region.
            bullet_data = np.zeros(4 * TRACKED_BULLETS)
            available_bullet_positions = np.arange(TRACKED_BULLETS)
            positions = np.random.choice(available_bullet_positions, min(len(obs['bullets']), TRACKED_BULLETS), False)
            for i, bullet in enumerate(obs['bullets']):
                if i >= TRACKED_BULLETS:
                    break
                bullet_data[4 * positions[i] : 4 * (positions[i] + 1)] = bullet

            # Position enemies randomly within the enemies region. 
            enemy_data = np.zeros(2 * TRACKED_ENEMIES)
            available_enemy_positions = np.arange(TRACKED_ENEMIES)
            positions = np.random.choice(available_enemy_positions, min(len(obs['targets']), TRACKED_ENEMIES), False)
            for i, enemy in enumerate(obs['targets']):
                if i >= TRACKED_ENEMIES:
                    break
                enemy_data[2 * positions[i] : 2 * (positions[i] + 1)] = enemy

            # Player data is always 2-dimensional
            player_data = obs['agent']
            
            # Wall data is always FIELDWIDTH x FIELDHEIGHT-dimensional
            wall_data = np.array(obs['map']).flatten()

            obs_arr = np.concatenate((bullet_data, enemy_data, player_data, wall_data))
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32)

            probs_shoot, probs_move = policy_net(obs_tensor)
            
            dist_shoot = Categorical(probs_shoot)
            action_shoot = dist_shoot.sample().item()
            logp_shoot = dist_shoot.log_prob(torch.tensor(action_shoot)).item()
            
            dist_move = Categorical(probs_move)
            action_move = dist_move.sample().item()
            logp_move = dist_move.log_prob(torch.tensor(action_move)).item()

            next_obs, reward, done, _, _ = env.step((action_move, action_shoot))

            obs_buf.append(obs_tensor)
            act_move_buf.append(action_move)
            act_shoot_buf.append(action_shoot)
            logp_move_buf.append(logp_move)
            logp_shoot_buf.append(logp_shoot)
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
        torch.stack(obs_buf, dim=0), 
        torch.tensor(np.array(act_move_buf), dtype=torch.int32), 
        torch.tensor(np.array(act_shoot_buf), dtype=torch.int32), 
        torch.tensor(np.array(logp_move_buf), dtype=torch.float32), 
        torch.tensor(np.array(logp_shoot_buf), dtype=torch.float32), 
        advantages, 
        torch.tensor(np.array(returns), dtype=torch.float32),
        traj_rewards
    )


def ppo(
    env, policy_lr=3e-4, value_lr=1e-3, epochs=100, steps_per_trajectory=2048, 
    gamma=0.99, lam=0.95, clip_ratio=0.2, target_kl=0.01, train_policy_iters=80, 
    train_value_iters=80, trajectories_per_epoch=10,
):
    if not isinstance(env, WiiTanks):
        raise ValueError('Invalid environment. Only WiiTanks is supported.')
    
    #environment state space dimension, hardcoded for WiiTanks because it's hard to find on the fly
    obs_dim = FIELDWIDTH * FIELDHEIGHT + 4 * TRACKED_BULLETS + 2 * TRACKED_ENEMIES + 2
    act_dim = 18 #(8 directions to shoot + 1 option to not shoot) + (8 directions to move + 1 option to not move)

    policy_net = PolicyNetwork(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    for epoch in range(epochs):
        obs_tensor, act_move_tensor, act_shoot_tensor, logp_old_move_tensor, logp_old_shoot_tensor, advantages, returns, rewards = rollout_trajectory_details(
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

            probs_shoot, probs_move = policy_net(obs_tensor)

            dist_shoot = Categorical(probs_shoot)
            logp_shoot = dist_shoot.log_prob(act_shoot_tensor)
            
            dist_move = Categorical(probs_move)
            logp_move = dist_move.log_prob(act_move_tensor)

            ratio = torch.exp(logp_move + logp_shoot - logp_old_move_tensor - logp_old_shoot_tensor)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            loss_policy = -(torch.min(ratio * advantages, clip_adv)).mean()

            kl = (logp_old_move_tensor + logp_old_shoot_tensor - logp_move - logp_shoot).mean().item()
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
