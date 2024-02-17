# from https://github.com/mnmldb/autonomous-drone-swarm
# adapted to be usable for ProB2-UI and SimB
import numpy as np
import gym
from gym import spaces
import random

import collections
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import sys
import warnings
warnings.filterwarnings("ignore")


class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]

class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True



class Grid(gym.Env):
    metadata = {'render.modes': ['console']}
    # action id
    XM = 0 # x minus
    XP = 1 # x plus
    YM = 2 # y minus
    YP = 3 # y plus

    def __init__(self, x_size=5, n_agents=2):
        super(Grid, self).__init__()

        # size of 2D grid
        self.x_size = x_size

        # number of agents
        self.n_agents = n_agents
        self.idx_agents = list(range(n_agents)) # [0, 1, 2, ..., n_agents - 1]

        # initialize the mapping status
        self.init_grid()

        # initialize the position of the agent
        self.init_agent()

        # define action space
        n_actions = 4 # LEFT, RIGHT, TOP, BOTTOM
        self.action_space = MultiAgentActionSpace([spaces.Discrete(n_actions) for _ in range(self.n_agents)])

        # define observation space (x and y coordinates)
        self.obs_low = np.zeros(2)
        self.obs_high = np.ones(2) * (self.x_size - 1)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self.obs_low, self.obs_high) for _ in range(self.n_agents)])

    def init_agent(self, initial_pos=None):
        self.agent_pos = []
        if initial_pos is not None:
            self.agent_pos = initial_pos
            for i in range(self.n_agents):
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
        else:
            for i in range(self.n_agents):
                agent_pos_x = random.randrange(0, self.x_size)
                agent_pos_y = random.randrange(0, self.x_size)
                self.agent_pos.append([agent_pos_x, agent_pos_y])
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1

        self.n_poi = self.x_size ** 2

    def init_grid(self):
        self.grid_status = np.zeros([self.x_size, self.x_size])

    def get_coverage(self):
        mapped_poi = (self.grid_status == 1).sum()
        return mapped_poi / self.n_poi

    def get_agent_obs(self):
        pos_copy = copy.deepcopy(self.agent_pos)

        return pos_copy

    def reset(self, initial_pos=None):
        self.init_grid()
        self.init_agent(initial_pos)

        return self.get_agent_obs()

    def step(self, action, i): # i: index of the drone
        # original position
        org_x  = copy.deepcopy(self.agent_pos[i][0])
        org_y  = copy.deepcopy(self.agent_pos[i][1])

        # move the agent
        if action == self.XM:
            self.agent_pos[i][0] -= 1
        elif action == self.XP:
            self.agent_pos[i][0] += 1
        elif action == self.YM:
            self.agent_pos[i][1] -= 1
        elif action == self.YP:
            self.agent_pos[i][1] += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # account for the boundaries of the grid (-2: out of the grid)
        if self.agent_pos[i][0] > self.x_size - 1 or self.agent_pos[i][0] < 0 or self.agent_pos[i][1] > self.x_size - 1 or self.agent_pos[i][1] < 0:
            self.agent_pos[i][0] = org_x
            self.agent_pos[i][1] = org_y

        # reward
        prev_status = self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]]
        if prev_status == 0:
            reward = 10
            self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
        else:
            reward = 0

        # done
        mapped_poi = (self.grid_status == 1).sum()
        if mapped_poi == self.n_poi:
            done = True
        else:
            done = False

        return self.get_agent_obs(), reward, done

    def close(self):
        pass

class QTables():
    def __init__(self, observation_space, action_space, eps_start=1, eps_end=0.1, gamma=0.9, r=0.99, lr=0.1):
        self.num_agents = len(observation_space)

        self.observation_space = observation_space
        self.observation_length = observation_space[0].shape[0]
        self.size = int(self.observation_space[0].high[0] - self.observation_space[0].low[0]) + 1

        self.action_space = action_space
        self.action_values = [0, 1, 2, 3] # corresponding to the column numbers in q table
        self.action_num = len(self.action_values) # 4

        self.eps = eps_start  # current epsilon
        self.eps_end = eps_end # epsilon lower bound
        self.r = r  # decrement rate of epsilon
        self.gamma = gamma  # discount rate
        self.lr = lr  # learning rate

        self.q_tables = []
        for agent_i in range(self.num_agents):
            self.q_tables.append(np.zeros([self.size**2, self.action_num]))

        self.q_tables_count = []
        for agent_i in range(self.num_agents):
            self.q_tables_count.append(np.zeros([self.size**2, self.action_num]))

    # support function: convert the fov to the unique row number in the q table
    def obs_to_row(self, obs_array):
        return obs_array[0] * self.size + obs_array[1]

    def get_action(self, obs, i):
        if np.random.rand() < self.eps:
            action = random.choice(self.action_values)
            greedy = False
        else:
            obs_row = self.obs_to_row(obs[i])
            action = np.argmax(self.q_tables[i][obs_row])
            greedy = True

        return action, greedy

    def update_eps(self):
        # update the epsilon
        if self.eps > self.eps_end: # lower bound
            self.eps *= self.r

    def train(self, obs, obs_next, action, reward, done, i):
        obs_row = self.obs_to_row(obs[i])
        obs_next_row = self.obs_to_row(obs_next[i])

        q_current = self.q_tables[i][obs_row][action] # current q value
        q_next_max = np.max(self.q_tables[i][obs_next_row]) # the maximum q value in the next state

        # update the q value
        if done:
            self.q_tables[i][obs_row][action] = q_current + self.lr * reward
        else:
            self.q_tables[i][obs_row][action] = q_current + self.lr * (reward + self.gamma * q_next_max - q_current)

        # update the count
        self.q_tables_count[i][obs_row][action] += 1


def train():
    # records for each episode
    time_steps = [] # number of time steps in total
    epsilons = [] # epsilon at the end of each episode
    greedy = [] # the ratio of greedy choices
    trajectory = []
    coverage = []

    q_class = []

    # parameters for training
    train_episodes = 100000
    size = 10
    n_agents = 2
    max_steps = size * 20

    # initialize the environment and the q tables
    env = Grid(x_size=size, n_agents=2)
    q = QTables(observation_space=env.observation_space, action_space=env.action_space, eps_start=1, eps_end=0.1, gamma=0.9, r=0.99, lr=0.1)

    # training
    for episode in range(train_episodes):
        env.reset(initial_pos=[[0, 0], [9, 9]])
        state = env.get_agent_obs()
        eps_tmp = q.eps

        for step in range(max_steps):
            action_order = random.sample(env.idx_agents, env.n_agents) # return a random order of the drone indice
            for i in action_order:
                action, greedy_tf = q.get_action(obs=state, i=i)
                next_state, reward, done = env.step(action, i=i)
                q.train(state, next_state, action, reward, done, i)

                if done:
                    break

                # update the observation
                state = next_state

        # record
        epsilons.append(eps_tmp)
        coverage.append(env.get_coverage())

        if episode % 1000 == 0:
            q_class.append(copy.deepcopy(q))

        # update epsilon
        q.update_eps()

        print(episode, epsilons[episode], coverage[episode])
        joblib.dump(q_class, "q_class_multi_fixed.txt", compress=3)
    return q

def get_drone_positions(obs):
    return "{" + "1 |-> ({0} |-> {1}), 2 |-> ({2} |-> {3})".format(obs[0][0], obs[0][1], obs[1][0], obs[1][1]) + "}"

# initialize the environment
n_agents = 2
env = Grid(x_size=10, n_agents=n_agents)

action_names = {
    0: "Move_Left",
    1: "Move_Right",
    2: "Move_Up",
    3: "Move_Down"
}

action_names_inv = {val: key for key, val in action_names.items()}

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train()
    else:
        model = joblib.load("q_class_multi_fixed.txt")[-1]
        while True:
            obs = env.reset()
            rewards = 0.0
            info = None
            done = False
            finished = False
            j = 0
            delta = 1000
            finished = (int(input("")) == 1)
            input("")
            print("$initialise_machine")
            print(0)
            print("drone_positions = {0}".format(get_drone_positions(obs)))
            print("false")

            while not done and not finished:
                for i in range(n_agents):
                    finished = int(input("")) == 1
                    if finished:
                        break

                    enabled_operations = input("")
                    operations_list = enabled_operations.split(",")

                    q_tables = model.q_tables
                    obs_row = model.obs_to_row(obs[i])
                    predictions = np.array(q_tables[i][obs_row])
                    action_order = (-predictions).argsort()

                    new_action = 0

                    for action in action_order:
                        if action_names.get(int(action)) in operations_list:
                            new_action = action
                            break

                    obs, rewards, done = env.step(int(new_action), i=i)
                    actionName = action_names.get(int(new_action))

                    print(actionName)
                    print(delta if i == n_agents - 1 else 0)
                    print("drone_positions = {0}".format(get_drone_positions(obs)))
                    print("true" if done else "false")