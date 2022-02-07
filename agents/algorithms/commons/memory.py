from collections import deque, namedtuple
import torch as T
import numpy as np
import random

class ReplayBuffer():

    def __init__(self, replay_size, batch_size):

        self.buffer_size = replay_size
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])

    def remember(self, state,action,reward,next_state,done):

        e = self.experiences(state,action,reward,next_state,done)
        self.replay_buffer.append(e)

    def remember_hsac(self, experience):

        
        self.replay_buffer.append(experience)

    def sample(self):

        experiences = random.sample(self.replay_buffer, self.batch_size)
        states = T.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = T.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = T.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = T.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = T.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states,actions,rewards,next_states,dones)

    def sample_hsac(self):

        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:

            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), np.array(r_lst), np.array(s_prime_lst), np.array(done_mask_lst)
