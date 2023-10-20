import gymnasium as gym
import json
import os
import sys

from stable_baselines3 import A2C
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
        objects.append("{0} |-> ({1} |-> {2}) |-> ({3} |-> {4})".format(
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


        model = A2C('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    gamma=0.9,  # Discount factor
                    verbose=1,
                    tensorboard_log=save_path)


        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=save_path,
            name_prefix="rl_model"
        )

        model.learn(int(20_000), callback=checkpoint_callback, tb_log_name="new_dqn", progress_bar=True)
        model.save(model_path)
    else:
        # Load pre-trained model
        model = A2C.load("models/new/trained_model")
        while True:
            obs = env.reset()[0]
            rewards = 0.0
            info = None
            done = False
            finished = False
            j = 0
            delta = 10
            finished = (int(input("")) == 1)
            print("")
            input("")
            print("$initialise_machine")
            print(0)
            print("x = {0}".format(get_x(obs)) + " & " + "y = {0}".format(get_y(obs)) + " & " + "v_x = {0}".format(get_v_x(obs)) + " & " + "v_y = {0}".format(get_v_y(obs)) + " & " + "angle = {0}".format(get_angle(obs)) + " & " + "v_angular = {0}".format(get_v_angular(obs)) + " & " + "left_leg_on_ground = {0}".format(get_left_leg_on_ground(obs)) + " & " + "right_leg_on_ground = {0}".format(get_right_leg_on_ground(obs)) + " & " + "surface = {0}".format(get_surface(env)))
            print("false")

            while not done:
                finished = int(input("")) == 1
                if finished:
                    break

                j = j + 1

                action, _states = model.predict(obs)

                print(action_names.get(int(action)))
                corrected_action = input("")
                corrected_action = action_names_inv.get(corrected_action)

                obs, rewards, done, truncated, info = env.step(int(corrected_action))

                actionName = action_names.get(int(corrected_action))

                print(actionName)
                print(delta)
                print("x = {0}".format(get_x(obs)) + " & " + "y = {0}".format(get_y(obs)) + " & " + "v_x = {0}".format(get_v_x(obs)) + " & " + "v_y = {0}".format(get_v_y(obs)) + " & " + "angle = {0}".format(get_angle(obs)) + " & " + "v_angular = {0}".format(get_v_angular(obs)) + " & " + "left_leg_on_ground = {0}".format(get_left_leg_on_ground(obs)) + " & " + "right_leg_on_ground = {0}".format(get_right_leg_on_ground(obs)) + " & " + "surface = {0}".format(get_surface(env)))
                print("true" if done else "false")


env.close()