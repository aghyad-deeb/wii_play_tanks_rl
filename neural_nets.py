import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from tqdm import tqdm
from tanks import WiiTanks
from settings import FIELDHEIGHT, FIELDWIDTH, TRACKED_BULLETS, TRACKED_ENEMIES
import matplotlib.pyplot as plt


four_layer = False

def main():
    # cart_env = gym.make("CartPole-v1")
    # ppo(cart_env)
    tanks_env = WiiTanks()
    x = 5
    policy_net, value_net, epoch_data = ppo(tanks_env, steps_per_trajectory=1000, trajectories_per_epoch=100, clip_ratio=0.2, walls=False, joint_model=False, x=x)
    torch.save(value_net, f'value_net_3layer_{x}')
    torch.save(policy_net, f'policy_net_3layer_{x}')
    np.save(f'epoch_data_3layer_{x}.npy', np.array(epoch_data))

    value_net = torch.load(f'value_net_3layer_{x}')
    policy_net = torch.load(f'policy_net_3layer_{x}')
    epoch_data = np.load(f'epoch_data_3layer_{x}.npy')

    epochs = np.arange(0, len(epoch_data)) #epoch zero indicates no training yet

    plt.errorbar(epochs, epoch_data[:, 0], yerr=epoch_data[:, 1], ecolor='blue', capsize=3, linestyle='None', markerfacecolor='black', markeredgecolor='black', marker='o', markersize=5)
    plt.xlabel('Training Epoch')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"rewards_{x}.png", dpi=300)
    plt.show()

    plt.plot(epochs, epoch_data[:, 2], 'b')
    plt.xlabel('Training Epoch')
    plt.ylabel('Win Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"win_ratio_{x}.png", dpi=300)
    plt.show()
    



class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1000):
        '''
        Neural network for learning a policy. Forward on this module takes in observation states and produces probability
        distributions over the action space.

        Args:
            input_dim: Dimensionality of the observation/state space vector that will be passed to the network.

            output_dim: Dimensionality of the action space.

            hidden_dim: Number of nodes in the two hidden layers.
        '''
        super(PolicyNetwork, self).__init__()
        self.joint_model = True if (output_dim == 81) else False
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        if four_layer:
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        '''
        pi(s)

        Given an observation/state tensor x, return corresponding probability distributions over the action space.
        '''
        if torch.any(torch.isinf(x)) or torch.any(torch.isinf(-x)):
            print('invalid input')
        x = self.fc1(x)
        x = torch.tanh(x)
        x = torch.tanh(self.fc2(x))
        if four_layer:
            x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        if self.joint_model:
            return torch.log_softmax(x, dim=-1)
        if len(x.shape) == 1:
            return torch.log_softmax(x[:9], dim=-1), torch.log_softmax(x[9:], dim=-1)
        else:
            return torch.log_softmax(x[:, :9], dim=-1), torch.log_softmax(x[:, 9:], dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=1000):
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
        if four_layer:
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        '''
        V(s)

        Given an observation/state tensor x, return corresponding value functions for each state.
        '''
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        if four_layer:
            x = torch.tanh(self.fc3(x))
        return self.fc4(x)


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



def rollout_trajectory_details(n, env, steps_per_trajectory, policy_net, value_net, gamma, lam, walls=True, joint_model=False, x=1):
    '''
    Gets summary statistics for `n` trajectories.
    '''
    with torch.no_grad():
        obs_buf = []
        act_move_buf = []
        act_shoot_buf = []
        logp_move_buf = []
        logp_shoot_buf = []
        ep_rews = []
        traj_rewards = []
        all_advantages = []
        all_returns = []
        num_won = 0

        for _ in tqdm(range(n)):
            obs = env.reset(level=x)[0]
            done = False
            traj_reward = 0
            count = 0
            for l in range(steps_per_trajectory):
                # Process observation before running network
                # Vectorize in order of bullets, enemies, player, walls

                # Position bullets randomly within the bullets region.
                bullet_data = np.zeros(4 * TRACKED_BULLETS)
                available_bullet_positions = np.arange(TRACKED_BULLETS)
                if len(obs['bullets']) != 0:
                    positions = np.random.choice(available_bullet_positions, min(len(obs['bullets'][0]), TRACKED_BULLETS), False)
                    for i, bullet in enumerate(obs['bullets'][0]):
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
                if walls:
                    wall_data = np.array(obs['map']).flatten()
                    obs_arr = np.concatenate((bullet_data, enemy_data, player_data, wall_data))
                else:
                    obs_arr = np.concatenate((bullet_data, enemy_data, player_data))

                obs_tensor = torch.tensor(obs_arr, dtype=torch.float32)

                if not joint_model:
                    probs_shoot, probs_move = policy_net(obs_tensor)

                    dist_shoot = Categorical(torch.exp(probs_shoot))
                    action_shoot = dist_shoot.sample().item()
                    logp_shoot = dist_shoot.log_prob(torch.tensor(action_shoot)).item()
                    
                    dist_move = Categorical(torch.exp(probs_move))
                    action_move = dist_move.sample().item()
                    logp_move = dist_move.log_prob(torch.tensor(action_move)).item()

                else:
                    probs_move = policy_net(obs_tensor)
                    
                    dist_move = Categorical(torch.exp(probs_move))
                    action = dist_move.sample().item()
                    logp_move = dist_move.log_prob(torch.tensor(action)).item()
                    logp_shoot = 0

                    action_move = action//9
                    action_shoot = action - 9 * action_move

                next_obs, reward, done, _, won = env.step((action_move, action_shoot))
                
                if won:
                    num_won += 1

                if done and l == 0:
                    break

                if not joint_model:
                    act_move_buf.append(action_move)
                else:
                    act_move_buf.append(action)

                obs_buf.append(obs_tensor)
                act_shoot_buf.append(action_shoot)
                logp_move_buf.append(logp_move)
                logp_shoot_buf.append(logp_shoot)
                ep_rews.append(reward)
                traj_reward += reward

                obs = next_obs

                count += 1

                if done:
                    break

            if count == 0:
                continue

            # Compute value and advantage targets
            traj_rewards.append(traj_reward/count)
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
        traj_rewards,
        num_won/n
    )


def ppo(
    env, policy_lr=1e-4, value_lr=1e-4, epochs=101, steps_per_trajectory=2048, 
    gamma=0.99, lam=0.95, clip_ratio=0.2, target_kl=0.01, train_policy_iters=100, 
    train_value_iters=100, trajectories_per_epoch=10, walls=True, joint_model=False, x=1
):
    if not isinstance(env, WiiTanks):
        raise ValueError('Invalid environment. Only WiiTanks is supported.')
    
    #environment state space dimension, hardcoded for WiiTanks because it's hard to find on the fly
    if walls:
        obs_dim = FIELDWIDTH * FIELDHEIGHT + 4 * TRACKED_BULLETS + 2 * TRACKED_ENEMIES + 2
    else:
        obs_dim = 4 * TRACKED_BULLETS + 2 * TRACKED_ENEMIES + 2

    if joint_model:
        act_dim = 81 #(8 directions to shoot + 1 option to not shoot) * (8 directions to move + 1 option to not move)
    else:
        act_dim = 18 #(8 directions to shoot + 1 option to not shoot) + (8 directions to move + 1 option to not move)

    # policy_net = PolicyNetwork(obs_dim, act_dim)
    # value_net = ValueNetwork(obs_dim)
    
    value_net = torch.load(f'value_net_3layer_{x-2}')
    policy_net = torch.load(f'policy_net_3layer_{x-2}')

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    epoch_data = []

    for epoch in range(epochs):
        obs_tensor, act_move_tensor, act_shoot_tensor, logp_old_move_tensor, logp_old_shoot_tensor, advantages, returns, rewards, win_ratio = rollout_trajectory_details(
            trajectories_per_epoch, env, steps_per_trajectory, policy_net, value_net, gamma, lam, walls, joint_model, x
        )

        epoch_data.append([np.mean(rewards), np.std(rewards, ddof=1), win_ratio])

        print(
            "Epoch: ", epoch, 
            "\tAverage Reward: ", np.mean(rewards), 
            "\tStandard Deviation: ", np.std(rewards, ddof=1),
            "\tWin Ratio: ", win_ratio,
        )

        # PPO Policy Update
        if not joint_model:
            for _ in range(train_policy_iters):
                policy_optimizer.zero_grad()

                probs_shoot, probs_move = policy_net(obs_tensor)

                # try:
                dist_shoot = Categorical(torch.exp(probs_shoot))
                logp_shoot = dist_shoot.log_prob(act_shoot_tensor)
                
                dist_move = Categorical(torch.exp(probs_move))
                logp_move = dist_move.log_prob(act_move_tensor)

                ratio = torch.exp(logp_move + logp_shoot - logp_old_move_tensor - logp_old_shoot_tensor)
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                loss_policy = -(torch.min(ratio * advantages, clip_adv)).mean()
                
                kl = (logp_old_move_tensor + logp_old_shoot_tensor - logp_move - logp_shoot).mean().item()
                if kl > target_kl:
                    break

                loss_policy.backward()
                policy_optimizer.step()

                # except ValueError as e:
                #     print(f'Got tensor of nans. Exception: {e}')

        else:
            for _ in range(train_policy_iters):
                policy_optimizer.zero_grad()

                probs = policy_net(obs_tensor)

                dist = Categorical(probs)
                logp = dist.log_prob(torch.exp(act_move_tensor))

                ratio = torch.exp(logp - logp_old_move_tensor)
                # print(ratio, clip_ratio)
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                loss_policy = -(torch.min(ratio * advantages, clip_adv)).mean()

                # print(loss_policy.detach())

                kl = (logp_old_move_tensor - logp).mean().item()
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


        if epoch % 10 == 1 or epoch == epochs - 1:
            torch.save(value_net, f'value_net_3layer_{x}')
            torch.save(policy_net, f'policy_net_3layer_{x}')
            np.save(f'epoch_data_3layer_{x}', np.array(epoch_data))


    env.close()
    return policy_net, value_net, np.array(epoch_data)


if __name__ == "__main__":
    main()
