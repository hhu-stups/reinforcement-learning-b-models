import gymnasium as gym
import json
import os
import sys
import socket
import json

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import warnings
warnings.filterwarnings("ignore")

save_base_path = "models"

env = gym.make('CliffWalking-v0')
env.reset()


action_names = {
    0: "Move_Up",
    1: "Move_Right",
    2: "Move_Down",
    3: "Move_Left"
}

action_names_inv = {val: key for key, val in action_names.items()}

def get_x(obs):
    return int(obs % 12)

def get_y(obs):
    return int(obs / 12)

def get_paths():
    global save_base_path
    if len(sys.argv) > 2:
        model_id = sys.argv[2]
    else:
        model_id = 'new'

    save_path = os.path.join(save_base_path, model_id)
    model_path = os.path.join(save_path, "trained_model")

    return save_path, model_path

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        save_path, model_path = get_paths()

        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.9,  # Discount factor
                    exploration_fraction=0.3,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log=save_path)

        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=save_path,
            name_prefix="rl_model"
        )

        model.learn(int(100_000), callback=checkpoint_callback, tb_log_name="new_dqn", progress_bar=True)
        model.save(model_path)
    else:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                port = int(sys.argv[1])
                client_socket.connect(("127.0.0.1", port))

                # Load pre-trained model
                model = DQN.load("models/new/trained_model")
                while True:
                    obs = env.reset()[0]
                    rewards = 0.0
                    info = None
                    done = False
                    finished = False
                    j = 0
                    delta = 1000

                    request = json.loads(client_socket.recv(1024).decode('utf-8'))
                    finished = (int(request['finished']) == 1)

                    response = json.dumps({
                        'op': '$initialise_machine',
                        'delta': 0,
                        'predicate': "x = {0}".format(get_x(obs)) + " & " + "y = {0}".format(get_y(obs)),
                        'done': 'false'
                    }) + "\n"
                    client_socket.sendall(response.encode('utf-8'))

                    while not done and not finished:
                        request = json.loads(client_socket.recv(1024).decode('utf-8'))
                        finished = (int(request['finished']) == 1)
                        if finished:
                            break

                        enabled_operations = request['enabledOperations']
                        operations_list = enabled_operations.split(",")

                        obs_tensor, _ = model.policy.obs_to_tensor(obs)
                        predictions = model.policy.q_net(obs_tensor)
                        action_order = (-predictions).argsort(dim=1)

                        new_action = 0

                        for action in action_order[0]:
                            if action_names.get(int(action)) in operations_list:
                                new_action = action
                                break

                        obs, rewards, done, truncated, info = env.step(int(new_action))
                        actionName = action_names.get(int(new_action))

                        response = json.dumps({
                            'op': actionName,
                            'delta': delta,
                            'predicate': "x = {0}".format(get_x(obs)) + " & " + "y = {0}".format(get_y(obs)),
                            'done': "true" if done else "false"
                        }) + "\n"
                        client_socket.sendall(response.encode('utf-8'))
        except Exception as e:
            print(f"Error: {e}")

env.close()