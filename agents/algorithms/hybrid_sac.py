import gym
import gym_platform
import numpy as np
import matplotlib.pyplot as plt
import random

from agents.agent import Agent
from agents.algorithms.commons.memory import ReplayBuffer
from agents.algorithms.models.sac_model import Policy, SoftQNetwork
from agents.algorithms.commons.utils import state_reduction, action_construction, to_torch_action, gym_to_buffer, to_gym_action

from wrappers.wrapper_gym_platform import ScaledStateWrapper, ScaledParameterisedActionWrapper, PlatformFlattenedActionWrapper

import torch as T
import torch.optim as optim

class HSAC(Agent):

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

        self.step_rate_eps = 0.03
        self.start_learning = 3000 #in steps

        self.alpha = 0.2 #Entropy Regul Coeff
        self.gamma = 0.9
        self.q_lr = 0.001
        self.policy_lr = 0.0001
        self.tau = 0.1


        self.replay_buffer_size = 10000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.replay_buffer_size, self.batch_size)

        self.target_network_frequency = 1 #TD3 tricks
        self.policy_frequency = 1

        self.device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")

        self.actor = Policy(self.k_f, self.action_size).to(self.device)
        self.q1 = SoftQNetwork(self.k_f, self.action_size).to(self.device)
        self.q2 = SoftQNetwork(self.k_f, self.action_size).to(self.device)
        self.q1_target = SoftQNetwork(self.k_f, self.action_size).to(self.device)
        self.q2_target = SoftQNetwork(self.k_f, self.action_size).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.values_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.q_lr)
        self.policy_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr)

        self._update_target_model()

    def train(self):
        """Test algorithm."""

        #Initiate variables
        episodes = 10000
        n_steps = 0
        cum_reward_lst = []
        mean_cum_rwd_lst = []

        for episode in range(episodes):

            # Reset environment
            state = self.env.reset()
            state = state_reduction(state, self.k_f, False)
            action = self._act(n_steps, state)

            # Initiate variables for each episode
            done = False
            episode_reward = 0

            while not done:

                state_, reward, done, _ = self.env.step(action)
                state_ = state_reduction(state_, self.k_f, False)
                self.memory.remember(state, gym_to_buffer(action), reward, state_, done)
                state = state_

                if len(self.memory.replay_buffer) > self.batch_size:

                    self._learn(n_steps)

                if self.target_network_frequency % 50 == 0 :

                    self._update_target_model()

                episode_reward += reward
                self.target_network_frequency += 1
                n_steps += 1
                action = self._act(n_steps,state)

            cum_reward_lst.append(episode_reward)

            if episode % 50 == 0:

                mean_r = np.mean(cum_reward_lst[-50:])
                print("Episode", episode,"/",episodes, "- Mean Reward:", round(mean_r,2))
                mean_cum_rwd_lst.append(mean_r)

        # Close the environment
        plt.plot(mean_cum_rwd_lst)
        plt.legend(["HSAC"])
        plt.xlabel("Episode")
        plt.ylabel("Mean Cumulated Reward")
        plt.show()

        self.env.close()

    def test(self):
        """Train algorithm."""

        print("There is no test yet for {} algorithm, you can only test it.".format(self.name))

    def _act(self, n_steps, state):

        if n_steps < self.start_learning:

            action = self.env.action_space.sample()
            action = gym_to_buffer(action)
            action = [action[0], action[1:]]

        else:

            state = T.Tensor(state).to(self.device)

            with T.no_grad():

                action_c, action_d, _, _, _ = self.actor.sample_normal(state)

            action = to_gym_action(action_c, action_d)


        return action

    def _update_target_model(self):

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):

            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):

            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def _learn(self, n_steps):

        states, actions, rewards, next_states, dones = self.memory.sample_hsac()
        states = T.Tensor(states).to(self.device)
        actions = T.Tensor(actions).to(self.device)
        rewards = T.Tensor(rewards).to(self.device)
        next_states = T.Tensor(next_states).to(self.device)
        dones = T.Tensor(dones).to(self.device)

        criterion = T.nn.MSELoss()

        with T.no_grad():

            next_state_actions_c, _, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = self.actor.sample_normal(next_states)
            q1_next_target = self.q1_target.forward(next_states, next_state_actions_c)
            q2_next_target = self.q2_target.forward(next_states, next_state_actions_c)
            min_qf_next_target = next_state_prob_d * (T.min(q1_next_target, q2_next_target) - self.alpha * next_state_prob_d * next_state_log_pi_c - self.alpha * next_state_log_pi_d)
            new_q_values = rewards + (self.gamma* (min_qf_next_target.sum(1)).view(-1) * (1 - dones))

        s_actions_c, s_actions_d = to_torch_action(actions)
        q1_a_values = self.q1.forward(states, s_actions_c).gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
        q2_a_values =self.q2.forward(states, s_actions_c).gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)

        q1_loss = criterion(q1_a_values, new_q_values)
        q2_loss = criterion(q2_a_values, new_q_values)

        q_loss = (q1_loss + q2_loss) / 2

        self.values_optimizer.zero_grad()
        q_loss.backward()
        self.values_optimizer.step()

        if n_steps % self.policy_frequency == 0:

            for _ in range(self.policy_frequency):

                actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.actor.sample_normal(states)
                q1_pi = self.q1.forward(states, actions_c)
                q2_pi = self.q2.forward(states, actions_c)
                min_q_pi = T.min(q1_pi, q2_pi)

                policy_loss_d = (prob_d * (self.alpha * log_pi_d - min_q_pi)).sum(1).mean()
                policy_loss_c = (prob_d * (self.alpha * prob_d * log_pi_c - min_q_pi)).sum(1).mean()
                policy_loss = policy_loss_d + policy_loss_c

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
