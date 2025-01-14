# Reinforcement Learning B Models

## Getting started

1. Install ProB2-UI in its latest nightly version as available here: https://github.com/hhu-stups/prob2_ui
2. Load the `RL_Project.prob2project` and start a formal model (`HighwayEnvironment2.mch` in this example)

![Start ProB2-UI](/images/Start_RL_Agent_1.png)

3. Open SimB in ProB2-UI

![Open SimB](/images/Start_RL_Agent_2.png)

4. Load a SimB simulation which starts an RL agent (extension `.py`). In this example, it is `HighwayEnvironment_Higher_Penalty_Collision.py`

![Load SimB Simulation](/images/Start_RL_Agent_3.png)

5. Start the RL agent as a SimB simulation by clicking on the `Start` button

![Play SimB Simulation](/images/Start_RL_Agent_4.png)


More information on SIMB including RL agents in SimB are available at: https://prob.hhu.de/w/index.php?title=SimB


## Simulation Scenarios

Here are some simulation scenarios for the HighwayEnvironment RL agent:

[Scenario Lane Left Collision](https://hhu-stups.github.io/highway-env-b-model/traces/Agent_Lane_Left_Collision)

[Scenario Slowing Down](https://hhu-stups.github.io/highway-env-b-model/traces/Agent_Slowing_Down)

[Scenario Base with Shield](https://hhu-stups.github.io/highway-env-b-model/traces/Agent_Scenario_Base_Shield)

[Scenario Higher Penalty with Shield](https://hhu-stups.github.io/highway-env-b-model/traces/Agent_Scenario_Higher_Penalty_Collision_Shield)


## Implementation overview

### Training of agents

Setup:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

The agent for the Highway Environment can be trained or tested via Command Line:

```bash
# To train a new model (save under models/highway_env):
python highway_agent.py train

# Loads the trained model and runs and renders episodes over its behaviour:
python highway_agent.py test
```

### Making an RL Agent compatible with SimB

First, make sure that the RL agent is available as a Python (`.py` file).
For the simulation to work, one has to apply the following steps:

1. Train a model of a Reinforcement Learning agent.

2. Create a formal B model (including safety shield) for the RL agent.

The operations represent the actions the RL agent can choose from. The formal model's state mainly represents the state of the environment. Safety shields are encoded by the operations' guards which are provided to the RL agent. Enabled operations are considered to be safe. Thus, the RL agent chooses the enabled operation/action with the highest predicted reward. The operations' substitutions model the desired behavior of the respective actions.

An example for the `FASTER` of the HighwayEnvironment is as follows:

```
FASTER = 
PRE
  EgoVehicle : dom(VehiclesVx) &
  ¬(∃v. (v ∈ PresentVehicles \ {EgoVehicle} ∧ VehiclesX(v) > 0.0 ∧ VehiclesX(v) < 45.0 ∧ 
  VehiclesY(v) < 3.5 ∧ VehiclesY(v) > -3.5))
THEN
  Crash :∈ BOOL ||
  PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles) ||
  VehiclesX :∈ Vehicles → R ||
  VehiclesY :∈ Vehicles → R ||
  VehiclesVx :| (VehiclesVx ∈ PresentVehicles → R ∧ 
             VehiclesVx(EgoVehicle) ≥ VehiclesVx’(EgoVehicle) - 0.05) ||
  VehiclesVy :∈ Vehicles → R ||
  VehiclesAx :| (VehiclesAx ∈ PresentVehicles → R ∧ VehiclesAx(EgoVehicle) ≥ -0.05) || 
  VehiclesAy :∈ Vehicles → R ||
  Reward :∈ R
END
```

Lines 2-4 shows the operation's guard which is used as safety shield.

Lines 6-15 shows the operation's substitution describing the desired behavior after executing `FASTER`.



3. Implement the mapping between the RL agent in Python and the formal B model. This includes the mapping of actions to operations, and the mapping of information from the RL agent, particularly the environment and observation, to the variables.

An example for the mapping of actions to operations is as follows:

```
action_names = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER"
}
```

Here, one can see that the numbers representing the actions in the RL agents are mapped to the corresponding operation names in the formal B model.


An example for a mapping to a variable in the formal B model is as follows:

```
def get_VehiclesX(obs):
    return "{{EgoVehicle |-> {0}, Vehicles2 |-> {1}, Vehicles3 |-> {2}, 
             Vehicles4 |-> {3},  Vehicles5 |-> {4}}}"
         .format(obs[0][1]*200, obs[1][1]*200, obs[2][1]*200, 
                  obs[3][1]*200, obs[4][1]*200) # Implemented manually
```

Remark: While the getter for the variable is generated by B2Program, the function for the mapping is implemented manually.


4. Implement necessary messages sent between the ProB animator and the RL agent. Simulation should be a while-loop which runs while the simulation has not finished.

- 1st message: Sent from ProB Animator: List of enabled operations
  - Meanwhile RL agent predicts enabled operation with highest reward
- 2nd message: Sent from RL agent: Name of chosen action/operation
- 3rd message: Sent from RL agent: Time until executing chosen action/operation
- 4th message: Sent from RL agent: Succeeding B state as a predicate
- 5th message: Sent from RL agent: Boolean flag describing whether simulation is finished

Example code (line 70 - 113; particularly 86 - 113): https://github.com/hhu-stups/reinforcement-learning-b-models/blob/main/HighwayEnvironment/HighwayEnvironment.py

To generate a RL agent for SimB, one can use the high-level code generator B2Program: https://github.com/favu100/b2program

Given a formal B model, B2Program generates an RL agent which loads a given trained model and execute the necessary steps. This includes the fourth step described before. The third step still has to be implemented; in this step, B2Program only generates the templates for the mappings which are then completed manually.



## Evaluation

This part describes the instructions to re-produce our evaluation results for the RL agent of the Highway Environment.
These are the Python and B machine files to start the `Base` and `Higher Penalty` Agents with and without shielding.

|                  | RL Agent                                         |
|------------------|--------------------------------------------------|
| `Base`           | `HighwayEnvironment_Base.py`                     |
| `Higher Penalty` | `HighwayEnvironment_Higher_Penalty_Collision.py` |

|             | Machine                   |
|-------------|---------------------------|
| `No Shield` | `HighwayEnvironment.mch`  |
| `Shield`    | `HighwayEnvironment2.mch` |

For example: In order to re-run the `Base` agent without `Shielding`, one would have to load `HighwayEnvironment.mch` with `HighwayEnvironment_Base.py` via SimB
As the RL agents' behavior are non-deterministic, particularly, probabilistic, the evaluation results might differ when re-running the agents for specific number of simulations.
For our evaluation results, we have saved the traces that were generated as shown below:


|                                | Saved Traces                                                                  |
|--------------------------------|-------------------------------------------------------------------------------|
| `Base` + `No Shield`           | Open `HighwayEnvironment.mch`, load `Traces_Base`                             |
| `Base` + `Shield`              | Open `HighwayEnvironment2.mch`, load `Traces_Base_Shield`                     |
| `Higher Penalty` + `No Shield` | Open `HighwayEnvironment.mch`, load `Traces_Higher_Penalty_Collision`         |
| `Higher Penalty` + `Shield`    | Open `HighwayEnvironment2.mch`, load `Traces_Higher_Penalty_Collision_Shield` |

For example: In order to re-run the results for `Base` + `No Shield` exacly with the same traces, it is necessary to open `HighwayEnvironment.mch` and load
the set of timed traces `Traces_Base` via SimB.


### Results

In the following, we show the results of applying SimB's statistical validation techniques to the RL agent of the Highway Environment.
The corresponding validation tasks are part of the ProB2-UI project `RL_Project.prob2project` in the corresponding machines.
They can be executed via the left-hand side of the SimB window.

#### Estimation of Average Values with Standard Deviation

| Metric            | Expression                                                                                                                               | Base without shield | Base with shield  | Higher Penalty without shield | Higher Penalty with shield |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------|---------------------|-------------------|-------------------------------|----------------------------|
| Episode Length    | *Inspection of Statistics*                                                                                                               | 38.85 +- 22.41      | 56.71 +- 11.46    | 53.02 +- 15.32                | 59.16 +- 5.54              |
| Velocity [m/s]    | Average of `RSQRT(VehiclesVy(EgoVehicle) * VehiclesVy(EgoVehicle) +                    VehiclesVx(EgoVehicle) * VehiclesVx(EgoVehicle))` | 23.37 +- 2.17       | 21.49 +- 0.94     | 21.14 +- 0.79                 | 20.95 +- 0.63              |
| Distance [m]      | Sum of `VehiclesVx(EgoVehicle)`                                                                                                          | 876.35 +- 477.62    | 1213.30 +- 244.48 | 1117.18 +- 321.71             | 1238.04 +- 122.12          |
| On Right Lane [s] | Sum of `IF VehiclesY(EgoVehicle) >= 7.0 THEN 1.0 ELSE 0.0 END`                                                                           | 31.69 +- 22.29      | 42.26 +- 20.51    | 47.07 +- 17.85                | 48.73 +- 17.52             |
| Total Reward      | Sum of `Reward`                                                                                                                          | 30.41 +- 17.39      | 42.88 +- 8.86     | 39.90 +- 11.77                | 44.20 +- 4.42              |

Example: In order to re-produce the results of Total Reward for `Base without shield`, one has to open `HighwayEnvironment.mch` and run the SimB validation task which computes the sum of `Reward` in `Traces_Base`.


#### Likelihood of Safety Properties in a 60 s Run

| Safety Property                                                                                                                                     | Predicate                                                                                                                                                                                    | Base without Shield | Base with Shield | Higher Penalty without Shield | Higher Penalty with Shield |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|------------------|-------------------------------|----------------------------|
| `SAF1`: The agent must avoid collisions with other vehicles.                                                                                        | `Crash = FALSE`                                                                                                                                                                              | 45.4 %              | 91.8 %           | 78.5 %                        | 97.4 %                     |
| `SAF2`: The agent must drive faster than 20 m/s.                                                                                                    | `Crash = FALSE =>    (RSQRT(VehiclesVy(EgoVehicle) * VehiclesVy(EgoVehicle) +           VehiclesVx(EgoVehicle) * VehiclesVx(EgoVehicle))    >= 20.0)`                                        | 93.4 %              | 91.4 %           | 76.9 %                        | 83.0 %                     |
| `SAF3`: The agent must drive slower than 30 m/s.                                                                                                    | `Crash = FALSE =>    (RSQRT(VehiclesVy(EgoVehicle) * VehiclesVy(EgoVehicle) +           VehiclesVx(EgoVehicle) * VehiclesVx(EgoVehicle))    <= 30.0)`                                        | 95.2 %              | 98.8 %           | 100.0 %                       | 100.0 %                    |
| `SAF4`: The agent should decelerate at a maximum of 5 m/s^2                                                                                         | `Crash = FALSE =>    (VehiclesAx(EgoVehicle) < 0.0 =>      RSQRT(VehiclesAy(EgoVehicle) * VehiclesAy(EgoVehicle) +            VehiclesAx(EgoVehicle) * VehiclesAx(EgoVehicle))      <= 5.0)` | 100.0 %             | 100.0 %          | 100.0 %                       | 100.0 %                    |
| `SAF5`: The agent should accelerate at a maximum of 5 m/s^2                                                                                         | `Crash = FALSE =>    (VehiclesAx(EgoVehicle) > 0.0 =>      RSQRT(VehiclesAy(EgoVehicle) * VehiclesAy(EgoVehicle) +            VehiclesAx(EgoVehicle) * VehiclesAx(EgoVehicle))      <= 5.0)` | 100.0 %             | 100.0 %          | 100.0 %                       | 100.0 %                    |
| `SAF6`: To each other vehicle, the agent should keep a lateral  safety distance of at least 2 m and a longitudinal safety distance of at least 10 m | `not(#v. (v : Vehicles \\ {EgoVehicle} &       VehiclesX(v) >= -15.0 & VehiclesX(v) <= 15.0 &       VehiclesY(v) >= -4.0 & VehiclesY(v) <= 4.0))`                                            | 6.4 %               | 49.2 %           | 41.6 %                        | 70.5 %                     |

Example: In order to re-produce the results of `SAF1` for `Base with shield`, one has to open `HighwayEnvironment2.mch` and run the SimB validation task which checks `Crash = FALSE` as an invariant for all simulated traces. The corresponding example is the first validation task in the following screenshot:

![Example: SimB Task](/images/Example_SimB_Task.png)


### Information about Formal Models, Environments, Trained Agents

- Formal Models
  - Autonomous Drone Swarm, Cliff Walking, Frozen Lake, Highway Environments (non-RSS models), Lunar Lander by Fabian Vu
  - Highway Environment RSS Models by Michael Leuschel
- Environments:
  - Cliff Walking, Frozen Lake, Lunar Lander from OpenAI Gymnasium
  - Highway Environment: Edouard Leurent. An Environment for Autonomous Driving Decision-Making. https://github.com/eleurent/highway-env
  - Autonomous Drone Swarm: Daisuke Nakanishi, Gurpreet Singh, Kshitij Chandna. Developing Autonomous Drone Swarms with Multi-Agent Reinforcement Learning for Scalable Post-Disaster Damage Assessment. https://github.com/mnmldb/autonomous-drone-swarm
- Training:
  - Autonomous Drone Swarm, Cliff Walking, Frozen Lake, Highway Environment (Refined Base, Refined Higher Penalty Collision, Refined Adversarial Agents), Lunar Lander by Fabian Vu
  - Highway Environment (Base and Higher Penalty Agents) by Jannik Dunkelau

