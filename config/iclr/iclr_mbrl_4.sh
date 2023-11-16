#!/usr/bin/env sh


python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our_50k --mode train --force --params agent_params/total_training_steps 50000 &
python main_ips.py --config ./config/iclr_mb_ubc.json --id iclr_mb_ubc_50k --mode train --force --params agent_params/total_training_steps 50000 &
python main_ips.py --config ./config/iclr_our.json --id iclr_our_mf_50k --mode train --force --params agent_params/total_training_steps 50000 &
python main_ips.py --config ./config/iclr_ubc.json --id iclr_ubc_mf_50k --mode train --force --params agent_params/total_training_steps 50000
