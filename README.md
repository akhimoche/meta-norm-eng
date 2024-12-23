# Overview
This project investigates solving the contracting problem in _sequential social dilemmas_ (SSDs) by generating and applying multi-agent conventions by means of a hierarchical LLM model with LLM agents. Specifically, we try to incentivise selfish agents to perform actions that promote social welfare by using our LLM network (STORM). Multi-agent reinforcement learning (MARL) requires many millions of iterations before it can possibly achieve complex joint-strategies, and so our goal is to be able to learn these behaviours more quickly than RL agent could. 

# Installation
To install, you first need to have **Python 3.11** installed. You can check this is the case by entering the following from the terminal, within the file directory.

```bash
pip install -r requirements.txt
```

