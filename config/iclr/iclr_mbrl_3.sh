#!/usr/bin/env sh


python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our_200k --mode train --force --params agent_params/total_training_steps 200000 &
python main_ips.py --config ./config/iclr_mb_ubc.json --id iclr_mb_ubc_200k --mode train --force --params agent_params/total_training_steps 200000 &
python main_ips.py --config ./config/iclr_our.json --id iclr_our_mf_200k --mode train --force --params agent_params/total_training_steps 200000 &
python main_ips.py --config ./config/iclr_ubc.json --id iclr_ubc_mf_200k --mode train --params agent_params/total_training_steps 200000 --force
