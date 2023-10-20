import gymnasium as gym
import json
import highway_env
from stable_baselines3 import DQN
import warnings
warnings.filterwarnings("ignore")

env = gym.make('highway-fast-v0')
env.reset()

# Load pre-trained model
model = DQN.load("models/highway_dqn/higher_col_penalty/trained_model")


action_names = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER"
}

action_names_inv = {
    "LANE_LEFT": 0,
    "IDLE": 1,
    "LANE_RIGHT": 2,
    "FASTER": 3,
    "SLOWER": 4
}

def get_Crash(obs, info):
    if info is None:
        return "FALSE"
    return "TRUE" if info["crashed"] else "FALSE"

def get_VehiclesX(obs):
    return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, Vehicles4 |-> {3}, Vehicles5 |-> {4}}}".format(obs[0][1]*200, obs[1][1]*200, obs[2][1]*200, obs[3][1]*200, obs[4][1]*200)

def get_VehiclesY(obs):
    return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, Vehicles4 |-> {3}, Vehicles5 |-> {4}}}".format(obs[0][2]*12, obs[1][2]*12, obs[2][2]*12, obs[3][2]*12, obs[4][2]*12)

def get_VehiclesVx(obs):
    return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, Vehicles4 |-> {3}, Vehicles5 |-> {4}}}".format(obs[0][3]*80, obs[1][3]*80, obs[2][3]*80, obs[3][3]*80, obs[4][3]*80)

def get_VehiclesVy(obs):
    return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, Vehicles4 |-> {3}, Vehicles5 |-> {4}}}".format(obs[0][4]*80, obs[1][4]*80, obs[2][4]*80, obs[3][4]*80, obs[4][4]*80)

def get_VehiclesAx(obs, prev_obs):
    if prev_obs is None:
        return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, Vehicles4 |-> {3}, Vehicles5 |-> {4}}}".format(0.0, 0.0, 0.0, 0.0, 0.0)
    return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, Vehicles4 |-> {3}, Vehicles5 |-> {4}}}".format(obs[0][3]*80 - prev_obs[0][3] * 80, obs[1][3]*80 - prev_obs[1][3] * 80, obs[2][3]*80 - prev_obs[2][3] * 80, obs[3][3]*80 - prev_obs[3][3] * 80, obs[4][3]*80 - prev_obs[4][3] * 80)

def get_VehiclesAy(obs, prev_obs):
    if prev_obs is None:
        return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, Vehicles4 |-> {3}, Vehicles5 |-> {4}}}".format(0.0, 0.0, 0.0, 0.0, 0.0)
    return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, Vehicles4 |-> {3}, Vehicles5 |-> {4}}}".format(obs[0][4]*80 - prev_obs[0][4] * 80, obs[1][4] * 80  - prev_obs[1][4]*80, obs[2][4]*80  - prev_obs[2][4]*80, obs[3][4]*80  - prev_obs[3][4]*80, obs[4][4]*80 -  - prev_obs[4][4]*80)

def get_Reward(obs, rewards):
    return rewards * 1.0


while True:
    obs = env.reset()[0]
    rewards = 0.0
    info = None
    prev_obs = None
    done = False
    finished = False
    j = 0
    delta = 1000
    input("")
    print("")
    input("")
    print("$initialise_machine")
    print(0)
    print("Crash = {0}".format(get_Crash(obs, info)) + " & " + "VehiclesX = {0}".format(get_VehiclesX(obs)) + " & " + "VehiclesY = {0}".format(get_VehiclesY(obs)) + " & " + "VehiclesVx = {0}".format(get_VehiclesVx(obs)) + " & " + "VehiclesVy = {0}".format(get_VehiclesVy(obs)) + " & " + "VehiclesAx = {0}".format(get_VehiclesAx(obs, prev_obs)) + " & " + "VehiclesAy = {0}".format(get_VehiclesAy(obs, prev_obs)) + " & " + "Reward = {0}".format(get_Reward(obs, rewards)))
    print("false")

    while not done:
        finished = int(input("")) == 1
        if finished:
            break

        j = j + 1
        action, _states = model.predict(obs)
        prev_obs = obs

        print(action_names.get(int(action)))
        corrected_action = input("")
        corrected_action = action_names_inv.get(corrected_action)

        obs, rewards, done, truncated, info = env.step(int(corrected_action))
        actionName = action_names.get(int(corrected_action))

        print(actionName)
        print(delta)
        print("Crash = {0}".format(get_Crash(obs, info)) + " & " + "VehiclesX = {0}".format(get_VehiclesX(obs)) + " & " + "VehiclesY = {0}".format(get_VehiclesY(obs)) + " & " + "VehiclesVx = {0}".format(get_VehiclesVx(obs)) + " & " + "VehiclesVy = {0}".format(get_VehiclesVy(obs)) + " & " + "VehiclesAx = {0}".format(get_VehiclesAx(obs, prev_obs)) + " & " + "VehiclesAy = {0}".format(get_VehiclesAy(obs, prev_obs)) + " & " + "Reward = {0}".format(get_Reward(obs, rewards)))
        print("true" if done else "false")

env.close()