import gymnasium as gym
import json
import os
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import warnings
warnings.filterwarnings("ignore")

save_base_path = "models"

env = gym.make('LunarLander-v2')
env.reset()


action_names = {
    0: "Do_Nothing",
    1: "Fire_Left",
    2: "Fire_Down",
    3: "Fire_Right"
}

action_names_inv = {val: key for key, val in action_names.items()}

def get_x(obs):
    return (obs[0] + 1.5)*800.0/3.0

def get_y(obs):
    return 200.0 - ((obs[1] + 1.5)*200.0/3.0)

def get_v_x(obs):
    return obs[2]

def get_v_y(obs):
    return obs[3]

def get_angle(obs):
    return obs[4]

def get_v_angular(obs):
    return obs[5]

def get_left_leg_on_ground(obs):
    if bool(obs[6]):
        return "TRUE"
    return "FALSE"

def get_right_leg_on_ground(obs):
    if bool(obs[7]):
        return "TRUE"
    return "FALSE"


def get_surface(env):
    fixtures = env.moon.fixtures
    objects = []
    for i in range(1, len(fixtures)):
        point1 = fixtures[i].shape.vertices[0]
        point2 = fixtures[i].shape.vertices[1]
        objects.append("{0} |-> (({1} |-> {2}) |-> ({3} |-> {4}))".format(
        i,
        point1[0] * 800.0/20.0,
        200.0 - point1[1] * 200.0/20.0,
        point2[0] * 800.0/20.0,
        200.0 - point2[1] * 200.0/20.0
        ))

    surface = "{" + ", ".join(objects) + "}"
    return surface

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

        model.learn(int(2_000_000), callback=checkpoint_callback, tb_log_name="new_dqn", progress_bar=True)
        model.save(model_path)
    else:
        # Load pre-trained model
        model = DQN.load("models/new/trained_model")
        while True:
            obs = env.reset()[0]
            rewards = 0.0
            info = None
            done = False
            finished = False
            j = 0
            delta = 10
            finished = (int(input("")) == 1)
            input("")
            print("$initialise_machine")
            print(0)
            print("x = {0}".format(get_x(obs)) + " & " + "y = {0}".format(get_y(obs)) + " & " + "v_x = {0}".format(get_v_x(obs)) + " & " + "v_y = {0}".format(get_v_y(obs)) + " & " + "angle = {0}".format(get_angle(obs)) + " & " + "v_angular = {0}".format(get_v_angular(obs)) + " & " + "left_leg_on_ground = {0}".format(get_left_leg_on_ground(obs)) + " & " + "right_leg_on_ground = {0}".format(get_right_leg_on_ground(obs)) + " & " + "surface = {0}".format(get_surface(env)))
            print("false")

            while not done and not finished:
                finished = int(input("")) == 1
                if finished:
                    break

                enabled_operations = input("")
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

                print(actionName)
                print(delta)
                print("x = {0}".format(get_x(obs)) + " & " + "y = {0}".format(get_y(obs)) + " & " + "v_x = {0}".format(get_v_x(obs)) + " & " + "v_y = {0}".format(get_v_y(obs)) + " & " + "angle = {0}".format(get_angle(obs)) + " & " + "v_angular = {0}".format(get_v_angular(obs)) + " & " + "left_leg_on_ground = {0}".format(get_left_leg_on_ground(obs)) + " & " + "right_leg_on_ground = {0}".format(get_right_leg_on_ground(obs)) + " & " + "surface = {0}".format(get_surface(env)))
                print("true" if done else "false")


env.close()