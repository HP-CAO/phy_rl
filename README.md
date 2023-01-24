# phy_rl
In this project, we explore physical knowledge affiliated reinforcement learning (RL) with a case study of an inverted pendulum.

## Description


## Installation

### Environment
This project is using the following settings:

- Ubuntu: 20.04 
- python: >3.6.5 

### Package

```
pip install -r requirement.txt
```

## Run

```
main_ddpg.py [-h] [--config CONFIG] [--generate_config] [--force]
                  [--params [PARAMS [PARAMS ...]]] [--mode MODE]
                  [--gpu] [--id RUN_ID][--weights PATH_TO_WEIGHTS] 

arguments:
  --config             Specifying different configuration .json files for different test.
  --generate_config    Generating default configuration .json files. 
  --force              Over-writting the previous run of same ID.
  --params             Over-writting params setting.
  --gpu                Enabling gpu device to speed up training. Training using CPU if not specified.   
  --mode MODE          Training or testing [train|test]
  --id                 Assigning an ID for the training/testing.
  --weights            Loading pretrained weights.    
```


Example_1: Generate configuration file

```
python main.py --generate_config
```

Example_2: Training/Testing

```
python main.py --config {PATH_TO_CONFIG_FILE} --mode {train|test} --id {RUN_NAME} --gpu --weights {PATH_TO_PRETRAINED_WEIGHTS}
```
