import gym
import gym_platform
import numpy as np
import matplotlib.pyplot as plt
import random

from agents.agent import Agent
from agents.algorithms.commons.memory import ReplayBuffer
from agents.algorithms.models.pdqn_model import QNetwork, ParamNetwork
from agents.algorithms.commons.utils import state_reduction, action_construction, to_torch_action, gym_to_buffer, to_gym_action

from wrappers.wrapper_gym_platform import ScaledStateWrapper, ScaledParameterisedActionWrapper, PlatformFlattenedActionWrapper

import torch as T
import torch.optim as optim
import torch.nn.functional as F

class PDQN(Agent):

    def create_algorithm(self):
        """Create algorithm."""
        self.env = gym.make('Platform-v0')
        self.env = ScaledStateWrapper(self.env)
        self.env = PlatformFlattenedActionWrapper(self.env)
        self.env = ScaledParameterisedActionWrapper(self.env)

        self.state_size = self.env.observation_space[0].shape[0]
        self.action_size = self.env.action_space[0].n

        self.rm_f = 5 # nb of features to remove from the state space
        self.k_f = self.state_size - self.rm_f #new state size with features removed

        self.step_rate_eps = 0.008

        self.gamma = 0.9
        self.lr = 0.00025

        self.epsilon = 1
        self.epsilon_min = 0.05

        self.replay_buffer_size = 10000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.replay_buffer_size, self.batch_size)

        self.target_network_frequency = 0

        self.device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")

        self.qnetwork = QNetwork(self.k_f, self.action_size).to(self.device)
        self.qnetwork_target = QNetwork(self.k_f, self.action_size).to(self.device)
        self.policynetwork = ParamNetwork(self.k_f, self.action_size).to(self.device)
        self.policynetwork_target = ParamNetwork(self.k_f, self.action_size).to(self.device)
        self.qoptimizer = optim.Adam(self.qnetwork.parameters(), lr= self.lr)
        self.policyoptimizer = optim.Adam(self.policynetwork.parameters(), lr= self.lr)

        self._update_target_model()

    def train(self):
        """Test algorithm."""

        #Initiate variables
        episodes = 10000
        cum_reward_lst = []
        mean_cum_rwd_lst = []

        for episode in range(episodes):

            state = self.env.reset()
            state = state_reduction(state, self.k_f, False)
            action = self._act(state)

            # Initiate variables for each episode
            done = False
            episode_reward = 0

            while not done:

                state_, reward, done, _ = self.env.step(action)
                state_ = state_reduction(state_, self.k_f,False)
                self.memory.remember(state, gym_to_buffer(action), reward, state_, done)
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

            action = self.env.action_space.sample()
            action = gym_to_buffer(action)
            action = [action[0], action[1:]]

        else:

            state = T.from_numpy(state).float().unsqueeze(0).to(self.device)

            with T.no_grad():

                action_param = self.policynetwork.forward(state)
                q_values = self.qnetwork.forward(state, action_param)

            action_param = action_param.view(-1)
            action_d = np.argmax(q_values)
            action = to_gym_action(action_param, action_d)

        return action

    def _update_target_model(self):

        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork.parameters()):

            target_param.data.copy_(local_param.data)

        for target_param, local_param in zip(self.policynetwork_target.parameters(), self.policynetwork.parameters()):

            target_param.data.copy_(local_param.data)

    def _update_epsilon(self,episode):

        ratio =  episode * self.step_rate_eps
        self.epsilon = 1/(1.1)**ratio

        if self.epsilon < self.epsilon_min:

            self.epsilon = self.epsilon_min

        return self.epsilon


    def _learn(self):

        states, actions, rewards, next_states, dones = self.memory.sample_hsac()
        states = T.Tensor(states).to(self.device)
        actions = T.Tensor(actions).to(self.device)
        rewards = T.Tensor(rewards).to(self.device)
        next_states = T.Tensor(next_states).to(self.device)
        dones = T.Tensor(dones).to(self.device)

        actions_c, actions_d = to_torch_action(actions)

        criterion = T.nn.MSELoss() #F.smooth_l1_loss
        self.qnetwork.train()
        self.policynetwork.train()
        self.qnetwork_target.eval()
        self.policynetwork_target.eval()


        with T.no_grad():

            next_params = self.policynetwork_target.forward(next_states)
            q_value_next = self.qnetwork_target(next_states, next_params)
            q_value_max_next = T.max(q_value_next, 1, keepdim=True)[0].squeeze()
            new_q_values = rewards + (self.gamma* q_value_max_next * (1-dones))

        q_predicted = self.qnetwork(states, actions_c).gather(1, actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)

        loss = criterion(q_predicted, new_q_values).to(self.device)
        self.qoptimizer.zero_grad()
        loss.backward()
        self.qoptimizer.step()

        with T.no_grad():

            action_params = self.policynetwork(states)

        action_params.requires_grad = True
        q_val = self.qnetwork(states, action_params)
        param_loss = -T.mean(T.sum(q_val, 1))
        self.policyoptimizer.zero_grad()
        param_loss.backward()
        self.policyoptimizer.step()
