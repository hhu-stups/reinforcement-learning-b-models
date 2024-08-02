import gymnasium as gym
import json
import highway_env
import copy
import sys
import socket
import json

from stable_baselines3 import DQN
import warnings
warnings.filterwarnings("ignore")

env = gym.make('highway-fast-v0')
env.reset()

# Load pre-trained model
model = DQN.load("models/highway_dqn/highest_col_penalty/trained_model")


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

def get_PresentVehicles(obs):
    vehicles = ["EgoVehicle"]
    for i in range(1,5):
        if int(obs[i][0]) == 1:
            vehicles.append("Vehicles{0}".format(i+1))
    return "{{{0}}}".format(", ".join(vehicles))

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

def read_line(socket):
    result = ''
    while True:
        data = socket.recv(1).decode('utf-8')
        if data == '\n':
            break
        result += data
    return result

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        port = int(sys.argv[1])
        client_socket.connect(("127.0.0.1", port))
        while True:
            obs = env.reset()[0]
            rewards = 0.0
            info = None
            prev_obs = None
            done = False
            finished = False
            delta = 1000

            request = json.loads(read_line(client_socket))
            finished = (int(request['finished']) == 1)

            response = json.dumps({
                'op': '$initialise_machine',
                'delta': 0,
                'predicate': "Crash = {0}".format(get_Crash(obs, info)) + " & " + "PresentVehicles = {0}".format(get_PresentVehicles(obs)) + " & " + "VehiclesX = {0}".format(get_VehiclesX(obs)) + " & " + "VehiclesY = {0}".format(get_VehiclesY(obs)) + " & " + "VehiclesVx = {0}".format(get_VehiclesVx(obs)) + " & " + "VehiclesVy = {0}".format(get_VehiclesVy(obs)) + " & " + "VehiclesAx = {0}".format(get_VehiclesAx(obs, prev_obs)) + " & " + "VehiclesAy = {0}".format(get_VehiclesAy(obs, prev_obs)) + " & " + "Reward = {0}".format(get_Reward(obs, rewards)),
                'done': 'false'
            }) + "\n"
            client_socket.sendall(response.encode('utf-8'))


            while not done and not finished:

                request = json.loads(read_line(client_socket))
                finished = (int(request['finished']) == 1)
                if finished:
                    break

                enabled_operations = request['enabledOperations']
                operations_list = enabled_operations.split(",")

                action, _states = model.predict(obs)
                new_action = action
                prev_obs = obs

                if not action_names.get(int(action)) in operations_list:
                    current_reward = -1.0
                    for forward_action in range(env.action_space.n):
                        if action == forward_action:
                            continue
                        if not action_names.get(forward_action) in operations_list:
                            continue
                        env_copy = copy.deepcopy(env)
                        _obs, action_reward, _done, _truncated, _info = env_copy.step(forward_action)
                        if action_reward > current_reward:
                            current_reward = action_reward
                            new_action = forward_action

                obs, rewards, done, truncated, info = env.step(int(new_action))
                actionName = action_names.get(int(new_action))

                response = json.dumps({
                    'op': actionName,
                    'delta': delta,
                    'predicate': "Crash = {0}".format(get_Crash(obs, info)) + " & " + "PresentVehicles = {0}".format(get_PresentVehicles(obs)) + " & " + "VehiclesX = {0}".format(get_VehiclesX(obs)) + " & " + "VehiclesY = {0}".format(get_VehiclesY(obs)) + " & " + "VehiclesVx = {0}".format(get_VehiclesVx(obs)) + " & " + "VehiclesVy = {0}".format(get_VehiclesVy(obs)) + " & " + "VehiclesAx = {0}".format(get_VehiclesAx(obs, prev_obs)) + " & " + "VehiclesAy = {0}".format(get_VehiclesAy(obs, prev_obs)) + " & " + "Reward = {0}".format(get_Reward(obs, rewards)),
                    'done': "true" if done else "false"
                }) + "\n"
                client_socket.sendall(response.encode('utf-8'))

    except Exception as e:
        print(f"Error: {e}")

env.close()