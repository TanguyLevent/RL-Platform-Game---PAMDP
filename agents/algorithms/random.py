from agents.agent import Agent
import gym
import gym_platform
import numpy as np
import matplotlib.pyplot as plt

class RandomAgent(Agent):


    def create_algorithm(self):
        """Create algorithm."""
        self.env = gym.make('Platform-v0')

    def train(self):
        """Train algorithm."""

        print("There is no train for {} algorithm, you can only test it.".format(self.name))

    def test(self):
        """Test algorithm."""

        #Initiate variables
        episodes = 10000
        cum_reward_lst = []
        mean_cum_rwd_lst = []

        for episode in range(episodes):

            # Reset environment
            state, _ = self.env.reset()
            action = self._act()

            # Initiate variables for each episode
            done = False
            episode_reward = 0

            while not done:

                state_, reward, done, _ = self.env.step(action)
                state, _ = state_
                action = self._act()

                episode_reward += reward

            cum_reward_lst.append(episode_reward)

            if episode % 50 == 0:

                mean_r = np.mean(cum_reward_lst[-50:])
                print("Episode", episode,"/",episodes, "- Mean Reward:", round(mean_r,2))
                mean_cum_rwd_lst.append(mean_r)

        # Close the environment
        plt.plot(mean_cum_rwd_lst)
        plt.legend(["Random"])
        plt.xlabel("Episode")
        plt.ylabel("Mean Cumulated Reward")
        plt.show()

        self.env.close()

    def _act(self):

        action = self.env.action_space.sample()

        return action
