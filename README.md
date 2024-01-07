# Reinforcement Learning B Models

## Getting started

1. Install ProB2-UI in its latest nightly version as available here: https://github.com/hhu-stups/prob2_ui
2. Load the `RL_Project.prob2project` and start a formal model (`HighwayEnvironment2.mch` in this example)

![Start ProB2-UI](/images/Start_RL_Agent_1.png)

3. Open SimB in ProB2-UI

![Open SimB](/images/Start_RL_Agent_2.png)

4. Load a SimB simulation which starts an RL agent (extension `.py`). In this example, it is `HighwayEnvironment_Higher_Penalty_Collision.py`

![Load SimB Simulation](/images/Start_RL_Agent_3.png)

5. Start the RL agent as a SimB simulation by clicking on the `Play` button

![Play SimB Simulation](/images/Start_RL_Agent_4.png)

## Scenarios

[Scenario Lane Left Collision](https://hhu-stups.github.io/highway-env-b-model/traces/Agent_Lane_Left_Collision)

[Scenario Slowing Down](https://hhu-stups.github.io/highway-env-b-model/traces/Agent_Slowing_Down)

[Scenario Base with Shield](https://hhu-stups.github.io/highway-env-b-model/traces/Agent_Scenario_Base_Shield)

[Scenario Higher Penalty with Shield](https://hhu-stups.github.io/highway-env-b-model/traces/Agent_Scenario_Higher_Penalty_Collision_Shield)

## Evaluation

This part describes the instructions to re-produce our evaluation results.
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

For eaxmple: In order to re-run the results for `Base` + `No Shield` exacly with the same traces, it is necessary to open `HighwayEnvironment.mch` and load
the set of timed traces `Traces_Base` via SimB.



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
