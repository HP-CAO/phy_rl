#!/usr/bin/env sh


python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our_75k --mode train --force --params agent_params/total_training_steps 75000 &
python main_ips.py --config ./config/iclr_mb_ubc.json --id iclr_mb_ubc_75k --mode train --force --params agent_params/total_training_steps 75000 &
python main_ips.py --config ./config/iclr_our.json --id iclr_our_mf_75k --mode train --force --params agent_params/total_training_steps 75000 &
python main_ips.py --config ./config/iclr_ubc.json --id iclr_ubc_mf_75k --mode train --force --params agent_params/total_training_steps 75000
