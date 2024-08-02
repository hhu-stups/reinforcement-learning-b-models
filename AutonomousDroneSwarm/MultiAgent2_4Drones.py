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
import socket
import json
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

    def __init__(self, x_size=5, y_size=5, n_agents=2, fov_x=3, fov_y=3):
        super(Grid, self).__init__()

        # size of 2D grid
        self.x_size = x_size
        self.y_size = y_size

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

        # define observation space (fielf of view)
        self.fov_x = fov_x # number of cells around the agent
        self.fov_y = fov_y # number of cells around the agent

        self.obs_low = -np.ones(4) * 2 # low -2: out of the grid
        self.obs_high = np.ones(4) # high 1: visited
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

        # iniqialize the stuck count
        self.stuck_counts = [0] * self.n_agents

    def init_grid(self):
        # initialize the mapping status
        ## -2: out of the grid
        ## -1: obstacle
        ## 0: POI that is not mapped
        ## 1: POI that is mapped
        self.grid_status = np.zeros([self.x_size, self.y_size])
        self.grid_counts = np.zeros([self.x_size, self.y_size])

        ## randomly set obstacles
        # n_obstacle = random.randrange(0, self.x_size * self.x_size * 0.2) # at most 20% of the grid
        n_obstacle = 0
        for i in range(n_obstacle):
            x_obstacle = random.randrange(1, self.x_size - 1)
            y_obstacle = random.randrange(1, self.y_size - 1)
            self.grid_status[x_obstacle, y_obstacle] = - 1
            self.grid_counts[x_obstacle, y_obstacle] = - 1

        # number of POI in the environment (0)
        self.n_poi = self.x_size * self.y_size - np.count_nonzero(self.grid_status)

    def get_coverage(self):
        mapped_poi = (self.grid_status == 1).sum()
        return mapped_poi / self.n_poi

    def get_agent_obs(self):
        self.agent_obs = []

        # observation for each agent
        for agent in range(self.n_agents):
            # default: out of the grid
            single_obs = -np.ones([self.fov_x, self.fov_y]) * 2
            for i in range(self.fov_x): # 0, 1, 2
                for j in range(self.fov_y): # 0, 1, 2
                    obs_x = self.agent_pos[agent][0] + (i - 1) # -1, 0, 1
                    obs_y = self.agent_pos[agent][1] + (j - 1) # -1, 0, 1
                    if obs_x >= 0 and obs_y >= 0 and obs_x <= self.x_size - 1 and obs_y <= self.y_size - 1:
                        single_obs[i][j] = copy.deepcopy(self.grid_status[obs_x][obs_y])
            single_obs_flat = single_obs.flatten() # convert matrix to list
            # extract the necessary cells
            xm = single_obs_flat[1]
            xp = single_obs_flat[7]
            ym = single_obs_flat[3]
            yp = single_obs_flat[5]
            single_obs_flat = np.array([xm, xp, ym, yp])
            self.agent_obs.append(single_obs_flat)
        return self.agent_obs

    def reset(self, initial_pos=None):
        # initialize the mapping status
        self.init_grid()
        # initialize the position of the agent
        self.init_agent(initial_pos)

        # check if the drones at initial positions are surrounded by obstacles
        while True:
            obs = self.get_agent_obs()
            obs_tf = []
            for i in range(self.n_agents):
                agent_obs_tf = obs[i][0] != 0 and obs[i][1] != 0 and obs[i][2] != 0 and obs[i][3] != 0
                obs_tf.append(agent_obs_tf)
            if any(obs_tf):
                self.init_grid()
                self.init_agent()
            else:
                break

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
        if self.agent_pos[i][0] > self.x_size - 1 or self.agent_pos[i][0] < 0 or self.agent_pos[i][1] > self.y_size - 1 or self.agent_pos[i][1] < 0:
            self.agent_pos[i][0] = org_x
            self.agent_pos[i][1] = org_y
            self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
            reward = 0
        else:
            # previous status of the cell
            prev_status = self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]]
            if prev_status == -1: # the new position is on the obstacle
                # go back to the original position
                self.agent_pos[i][0] = org_x
                self.agent_pos[i][1] = org_y
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0
            elif prev_status == 0:
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                self.grid_status[self.agent_pos[i][0], self.agent_pos[i][1]] = 1
                reward = 10
            elif prev_status == 1:
                self.grid_counts[self.agent_pos[i][0], self.agent_pos[i][1]] += 1
                reward = 0

        # update the stuck count
        if org_x == self.agent_pos[i][0] and org_y == self.agent_pos[i][1]: # stuck
            self.stuck_counts[i] += 1
        else:
            self.stuck_counts[i] = 0

        # are we map all cells?
        mapped_poi = (self.grid_status == 1).sum()
        done = bool(mapped_poi == self.n_poi)

        return self.get_agent_obs(), reward, done

    def close(self):
        pass

# multi-agent setting
# each agent has an individual q table

class QTables():
    def __init__(self, observation_space, action_space, eps_start=1, eps_end=0.1, gamma=0.9, r=0.99, lr=0.1):
        self.num_agents = len(observation_space)

        self.observation_space = observation_space
        self.observation_values = [-2, -1, 0, 1]
        self.observation_num = len(self.observation_values) # 3
        self.observation_length = observation_space[0].shape[0] # field of view

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
            self.q_tables.append(np.random.rand(self.observation_num**self.observation_length, self.action_num))

        self.q_table_counts = []
        for agent_i in range(self.num_agents):
            self.q_table_counts.append(np.zeros([self.observation_num**self.observation_length, self.action_num]))

    # support function: convert the fov to the unique row number in the q table
    def obs_to_row(self, obs_array):
        obs_shift = map(lambda x: x + 2, obs_array) # add 1 to each element
        obs_power = [v * (self.observation_num ** i) for i, v in enumerate(obs_shift)] # apply exponentiation to each element
        return sum(obs_power) # return the sum (results are between 0 and 256)

    def softmax(self, a):
        # deal with overflow
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def get_action(self, observations, agent_i, stuck_counts, max_stuck, e_greedy=True, softmax=False):
        # convert the observation to a row number
        obs_row = self.obs_to_row(observations[agent_i])
        if stuck_counts[agent_i] >= max_stuck: # random action to avoid stuck
            action = random.choice(self.action_values)
            greedy = False
            action_value = self.q_tables[agent_i][obs_row][action]
        elif e_greedy: # epsilon greedy for training (e_greedy=True)
            if np.random.rand() < self.eps:
                action = random.choice(self.action_values)
                greedy = False
                action_value = self.q_tables[agent_i][obs_row][action]
            else:
                action = np.argmax(self.q_tables[agent_i][obs_row])
                greedy = True
                action_value = self.q_tables[agent_i][obs_row][action]
        elif softmax: # (e_greedy=False and softmax=True)
            p = self.softmax(self.q_tables[agent_i][obs_row])
            action = np.random.choice(np.arange(self.action_num), p=p)
            greedy = False
            action_value = self.q_tables[agent_i][obs_row][action]
        else: # all greedy choices for testing performance
            action = np.argmax(self.q_tables[agent_i][obs_row])
            greedy = True
            action_value = self.q_tables[agent_i][obs_row][action]

        return action, greedy, action_value

    def update_eps(self):
        # update the epsilon
        if self.eps > self.eps_end: # lower bound
            self.eps *= self.r

    def train(self, obs, obs_next, action, reward, done, agent_i):
        obs_row = self.obs_to_row(obs[agent_i])
        obs_next_row = self.obs_to_row(obs_next[agent_i])
        act_col = action

        q_current = self.q_tables[agent_i][obs_row][act_col] # current q value
        q_next_max = np.max(self.q_tables[agent_i][obs_next_row]) # the maximum q value in the next state

        # update the q value
        if done:
            self.q_tables[agent_i][obs_row][act_col] = q_current + self.lr * reward
        else:
            self.q_tables[agent_i][obs_row][act_col] = q_current + self.lr * (reward + self.gamma * q_next_max - q_current)

        # inclement the corresponding count
        self.q_table_counts[agent_i][obs_row][act_col] += 1

def train():
    # ===================================================================================================
    # Training: 1 drone
    # ===================================================================================================

    # records for each episode
    time_steps = [] # number of time steps in total
    epsilons = [] # epsilon at the end of each episode
    greedy = [] # the ratio of greedy choices
    coverage = [] # the ratio of visited cells at the end
    speed = [] # number of time steps to cover decent amount of cells
    sum_q_values = [] # sum of q-values
    results_mapping = [] # mapping status
    results_count = [] # count status
    total_reward = []
    total_action_values = []
    total_greedy_action_values = []

    q_class = []

    coverage_threshold = 0.90
    max_stuck = 100000

    # parameters for training
    train_episodes = 25000
    max_steps = 10 * 10 * 2

    # initialize the environment and the q tables
    env = Grid(x_size=10, y_size=10, n_agents=4, fov_x=3, fov_y=3)
    q = QTables(observation_space=env.observation_space, action_space=env.action_space, eps_start=1, eps_end=0.1, gamma=0.9, r=0.99, lr=0.01)

    # training
    for episode in range(train_episodes):
        state = env.reset()
        state = [arr.astype('int') for arr in state] # convert from float to integer
        eps_tmp = q.eps

        greedy_count = [0] * env.n_agents
        coverage_track = True
        epi_reward = [0] * env.n_agents
        epi_action_value = [0] * env.n_agents
        epi_greedy_action_value = [0] * env.n_agents

        for step in range(max_steps):
            action_order = random.sample(env.idx_agents, env.n_agents) # return a random order of the drone indices
            for agent_i in action_order:
                action, greedy_tf, action_value = q.get_action(observations=state, agent_i=agent_i, stuck_counts=env.stuck_counts, max_stuck=max_stuck, e_greedy=True, softmax=False)
                next_state, reward, done = env.step(action, agent_i)
                next_state = [arr.astype('int') for arr in next_state] # convert from float to integer
                q.train(state, next_state, action, reward, done, agent_i)

                epi_reward[agent_i] += reward
                greedy_count[agent_i] += greedy_tf * 1
                epi_action_value[agent_i] += action_value
                epi_greedy_action_value[agent_i] += action_value * greedy_tf

                if done:
                    break

                # update the observation
                state = next_state

            # check if decent amoung of cells are visited
            current_coverage = env.get_coverage()
            if current_coverage >= coverage_threshold and coverage_track:
                speed.append(step)
                coverage_track = False

            # check if the task is completed
            if done:
                time_steps.append(step)
                break
            elif step == max_steps - 1:
                time_steps.append(step)
                if coverage_track:
                    speed.append(np.nan)

        # record
        time_steps.append(step + 1)
        epsilons.append(eps_tmp)
        coverage.append(env.get_coverage())
        greedy.append(list(map(lambda x: x / (step + 1), greedy_count)))
        sum_q_values.append([q.q_tables[0].sum()])
        results_mapping.append(env.grid_status)
        results_count.append(env.grid_counts)
        total_reward.append(epi_reward)
        total_action_values.append(epi_action_value)
        total_greedy_action_values.append(epi_greedy_action_value)

        if episode % 1000 == 0:
            q_class.append(copy.deepcopy(q))

        # update epsilon
        q.update_eps()

        print('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices　(%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Sum of Q-Values: {7:.1f},    Total Reward: {8}'\
              .format(episode+1, eps_tmp, step+1, np.mean(greedy[episode]), coverage[episode], coverage_threshold * 100, speed[episode], sum_q_values[episode][0], np.mean(total_reward[episode])))
        joblib.dump(q_class, "q_class_multi_fixed_fov_4.txt", compress=3)
    return q


def get_drone_positions(obs, agent_pos):
    return "{" + "1 |-> ({0} |-> {1}), 2 |-> ({2} |-> {3}), 3 |-> ({4} |-> {5}), 4 |-> ({6} |-> {7})".format(agent_pos[0][0], agent_pos[0][1], agent_pos[1][0], agent_pos[1][1], agent_pos[2][0], agent_pos[2][1], agent_pos[3][0], agent_pos[3][1]) + "}"

# initialize the environment
n_agents = 4
env = Grid(x_size=10, y_size=10, n_agents=n_agents)

action_names = {
    0: "Move_Left",
    1: "Move_Right",
    2: "Move_Up",
    3: "Move_Down"
}

action_names_inv = {val: key for key, val in action_names.items()}

def read_line(socket):
    result = ''
    while True:
        data = socket.recv(1).decode('utf-8')
        if data == '\n':
            break
        result += data
    return result

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train()
    else:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                port = int(sys.argv[1])
                client_socket.connect(("127.0.0.1", port))

                model = joblib.load("q_class_multi_fixed_fov_4.txt")[-1]
                while True:
                    obs = env.reset()
                    agent_pos = env.agent_pos
                    rewards = 0.0
                    info = None
                    done = False
                    finished = False
                    j = 0
                    delta = 1000

                    request = json.loads(read_line(client_socket))
                    finished = (int(request['finished']) == 1)

                    response = json.dumps({
                        'op': '$initialise_machine',
                        'delta': 0,
                        'predicate': "drone_positions = {0}".format(get_drone_positions(obs, agent_pos)),
                        'done': 'false'
                    }) + "\n"
                    client_socket.sendall(response.encode('utf-8'))

                    while not done and not finished:
                        for i in range(n_agents):
                            request = json.loads(read_line(client_socket))
                            finished = (int(request['finished']) == 1)
                            if finished:
                                break

                            enabled_operations = request['enabledOperations']
                            operations_list = enabled_operations.split(",")

                            q_tables = model.q_tables
                            obs_row = model.obs_to_row(obs[i])
                            predictions = np.array(q_tables[i][int(obs_row)])
                            action_order = (-predictions).argsort()

                            new_action = 0

                            for action in action_order:
                                if action_names.get(int(action)) in operations_list:
                                    new_action = action
                                    break

                            obs, rewards, done = env.step(int(new_action), i=i)
                            actionName = action_names.get(int(new_action))

                            response = json.dumps({
                                'op': actionName,
                                'delta': delta if i == n_agents - 1 else 0,
                                'predicate': "drone_positions = {0}".format(get_drone_positions(obs, agent_pos)),
                                'done': "true" if done else "false"
                            }) + "\n"
                            client_socket.sendall(response.encode('utf-8'))

                            if done:
                                break
        except Exception as e:
            print(f"Error: {e}")