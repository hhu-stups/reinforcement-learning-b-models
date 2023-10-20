# Implementation overview

## Training of agents

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
