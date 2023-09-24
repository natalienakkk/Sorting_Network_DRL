import numpy as np
from gym import spaces
from stable_baselines3 import DQN
import gym
import matplotlib.pyplot as plt
import time
import torch
from stable_baselines3.common.callbacks import BaseCallback

class SortingNetworkEnv(gym.Env):
    def __init__(self, num_elements, max_swaps):
        super(SortingNetworkEnv, self).__init__()

        self.num_elements = num_elements
        self.action_space = spaces.Discrete(num_elements - 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_elements,))
        self.comparators_used = 0
        self.max_swaps = max_swaps


    def reset(self):
        self.state = np.random.choice([0, 1], self.num_elements)
        self.steps_taken = 0
        self.comparators_used = 0
        self.actions_taken = []
        print("Initial elements:", self.state)
        return self.state.copy()

    def step(self, action):
        self.steps_taken += 1
        self.comparators_used += 1

        if action == self.num_elements - 1:
            return self.state.copy(), -10, False, {}

        current_out_of_order = self._get_num_out_of_order_pairs()
        i, j = action, action + 1
        self.state[i], self.state[j] = self.state[j], self.state[i]  # Perform the swap
        new_out_of_order = self._get_num_out_of_order_pairs()
        self.actions_taken.append(action)

        # Start with the reward based on out-of-order pairs
        reward = max(current_out_of_order - new_out_of_order, new_out_of_order-current_out_of_order)


        # Adjust rewards based on overall sorting progress
        if new_out_of_order < current_out_of_order:
            reward += 10
        elif new_out_of_order == current_out_of_order:
            reward += -3
        else:
            reward += -5

        # Final checks
        flag = 0
        if self._is_sorted():
            reward += self.num_elements * 10
            flag = 1

        if self._is_sorted() and env.comparators_used == env.max_swaps and flag == 1:
            reward += self.num_elements * 10

        if env.comparators_used == env.max_swaps and not env._is_sorted():
            reward -= env.num_elements * 10

        done = self._is_sorted()

        return self.state.copy(), reward, done, {}


    def _is_sorted(self):
        return np.all(self.state[:-1] <= self.state[1:])

    def _get_num_out_of_order_pairs(self):
        return np.sum(self.state[:-1] > self.state[1:])

env = SortingNetworkEnv(num_elements=16, max_swaps=61)
policy_kwargs = dict(net_arch=[256, 256])

#set parameters :
target_update_interval=1000
learning_rate = 0.001
gamma = 0.99
buffer_size = 10000

# model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, target_update_interval=target_update_interval,
#      exploration_initial_eps=1.0, exploration_fraction=0.3, gamma=gamma, buffer_size=buffer_size,
#      learning_rate=learning_rate, verbose=1)

model = DQN(
    'MlpPolicy',
    env,
    policy_kwargs=policy_kwargs,target_update_interval=1000,
    verbose=1,
    exploration_initial_eps=1.0,
    exploration_fraction=0.3,
    gamma=0.95,
    learning_rate=lambda x: 0.001 * (1 - x)
)

class QValueLogger(BaseCallback):
    def __init__(self, check_state, check_action, verbose=0):
        super(QValueLogger, self).__init__(verbose)
        self.q_values = []
        self.check_state = check_state
        self.check_action = check_action

    def _on_step(self) -> bool:
        obs_tensor = torch.tensor(self.check_state).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            q_values = self.model.policy.q_net(obs_tensor)
            q_value = q_values[0][self.check_action].item()
            self.q_values.append(q_value)
        return True

# Let's say you want to monitor Q-value for the initial state and action 0
selected_state = env.reset()
selected_action = 0
callback = QValueLogger(selected_state, selected_action)

model.learn(total_timesteps=100000, callback=callback)

#After training, you can plot Q-values.
plt.plot(callback.q_values)
plt.xlabel("Training Steps")
plt.ylabel("Q-Value for Selected (State, Action)")
plt.show()

#model.learn(total_timesteps=100000)
model.save("dqn_sorting_network")
model = DQN.load("dqn_sorting_network")

num_episodes = 10
total_rewards = 0
rewards_list = []  # A graph of the average reward per agent step.
episode_rewards = [] # A graph of the average reward per episode
episode_lengths = []
total_steps_taken = 0
episode_comparators = []  # Store number of comparators for each episode

start_time = time.time()
for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        rewards_list.append(episode_reward)

    episode_rewards.append(episode_reward)
    episode_comparators.append(env.comparators_used)
    total_steps_taken += env.steps_taken


    print(f"Reward for episode {episode + 1}: {episode_reward}")
    print("Sorted elements:", obs)
    #print(f"Comparators used for episode {episode + 1}: {env.comparators_used}")
    total_rewards += episode_reward

print("During Learning: ")
min_comparators = min(episode_comparators)
convergence_episode = episode_comparators.index(min_comparators)
print(f"The agent converged in episode {convergence_episode + 1}")

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime during learning: {runtime:.2f} seconds")

average_reward = total_rewards / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")

# Calculate the average episode length
average_episode_length = total_steps_taken / num_episodes
print(f"Average episode length over {num_episodes} episodes: {average_episode_length:.2f}")

#graph of the average reward per episode
cumulative_rewards = np.cumsum(episode_rewards)
average_rewards_per_episode = cumulative_rewards / (np.arange(len(episode_rewards)) + 1)
plt.figure(figsize=(10, 5))
plt.plot(average_rewards_per_episode, label="Average Reward")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Average Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()

# cumulative_rewards = np.cumsum(rewards_list)
# average_rewards_per_step = cumulative_rewards / (np.arange(len(rewards_list)) + 1)
#graph of the average reward per step
plt.figure(figsize=(10, 5))
plt.plot(rewards_list)
#plt.plot(average_rewards_per_step, label="Average Reward")
plt.xlabel('Timesteps')
plt.ylabel('Average Reward')
plt.title('Average Reward per Agent Step')
plt.show()

