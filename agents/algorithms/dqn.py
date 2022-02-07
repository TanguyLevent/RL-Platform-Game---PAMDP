import gym
import gym_platform
import numpy as np
import matplotlib.pyplot as plt
import random

from agents.agent import Agent
from agents.algorithms.commons.memory import ReplayBuffer
from agents.algorithms.models.dqn_model import QNetwork
from agents.algorithms.commons.utils import state_reduction, action_construction

import torch as T
import torch.optim as optim

class DQN(Agent):

    def create_algorithm(self):
        """Create algorithm."""
        self.env = gym.make('Platform-v0')

        self.state_size = self.env.observation_space[0].shape[0]
        self.action_size = self.env.action_space[0].n

        self.rm_f = 5 # nb of features to remove from the state space
        self.k_f = self.state_size - self.rm_f #new state size with features removed

        self.step_rate_eps = 0.03

        self.gamma = 0.9
        self.lr = 0.00025

        self.epsilon = 1
        self.epsilon_min = 0.05

        self.replay_buffer_size = 10000
        self.batch_size = 32
        self.memory = ReplayBuffer(self.replay_buffer_size, self.batch_size)

        self.target_network_frequency = 0

        self.device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")

        self.qnetwork = QNetwork(self.k_f, self.action_size).to(self.device)
        self.qnetwork_target = QNetwork(self.k_f, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr= self.lr)

        self._update_target_model()

    def train(self):
        """Test algorithm."""

        #Initiate variables
        episodes = 10000
        cum_reward_lst = []
        mean_cum_rwd_lst = []

        for episode in range(episodes):

            # Reset environment
            state = self.env.reset()
            state = state_reduction(state, self.k_f)
            action = self._act(state)

            # Initiate variables for each episode
            done = False
            episode_reward = 0

            while not done:

                action_c = action_construction(action, self.env.action_space.sample())
                state_, reward, done, _ = self.env.step(action_c)
                state_ = state_reduction(state_, self.k_f)
                self.memory.remember(state, action, reward, state_, done)
                state = state_

                if len(self.memory.replay_buffer) > self.batch_size:

                    self._learn()

                if self.target_network_frequency % 200 == 0 :

                    self._update_target_model()

                episode_reward += reward
                self.target_network_frequency += 1
                action = self._act(state)
                self._update_epsilon(episode)

            cum_reward_lst.append(episode_reward)

            if episode % 50 == 0:

                mean_r = np.mean(cum_reward_lst[-50:])
                print("Episode", episode,"/",episodes, "- Exploration rate:", round(self.epsilon,2) ,"- Mean Reward:", round(mean_r,2))
                mean_cum_rwd_lst.append(mean_r)

        # Close the environment
        plt.plot(mean_cum_rwd_lst)
        plt.legend(["DQN"])
        plt.xlabel("Episode")
        plt.ylabel("Mean Cumulated Reward")
        plt.show()

        self.env.close()

    def test(self):
        """Train algorithm."""

        print("There is no test yet for {} algorithm, you can only test it.".format(self.name))

    def _act(self, state):

        self.greedy = np.random.rand()

        if self.greedy <= self.epsilon:

            action = np.random.choice(self.action_size)

        else:

            state = T.from_numpy(state).float().unsqueeze(0).to(self.device)

            with T.no_grad():
                action_values = self.qnetwork(state)

            action = np.argmax(action_values.cpu().data.numpy())


        return action

    def _update_target_model(self):

        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork.parameters()):

            target_param.data.copy_(local_param.data)

    def _update_epsilon(self,episode):

        ratio =  episode * self.step_rate_eps
        self.epsilon = 1/(1.1)**ratio

        if self.epsilon < self.epsilon_min:

            self.epsilon = self.epsilon_min

        return self.epsilon


    def _learn(self):

        experience = self.memory.sample()
        states, actions, rewards, next_states, dones = experience

        criterion = T.nn.MSELoss()
        self.qnetwork.train()
        self.qnetwork_target.eval()

        q_predicted_targets = self.qnetwork(states).gather(1,actions)

        with T.no_grad():

            labels_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        new_q_values = rewards + (self.gamma* labels_next*(1-dones))

        loss = criterion(q_predicted_targets,new_q_values).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
