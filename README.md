# Reinforcement Learning B Models

## Getting started

![Start ProB2-UI](/images/Start_RL_Agent_1.png)

![Open SimB](/images/Start_RL_Agent_2.png)

![Load SimB Simulation](/images/Start_RL_Agent_3.png)

![Play SimB Simulation](/images/Start_RL_Agent_4.png)

## Scenarios

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
