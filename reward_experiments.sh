#!/usr/bin/env sh

python main.py --config ./config/reward_function/reward_all.json --id reward_all --mode train &
python main.py --config ./config/reward_function/reward_dis.json --id reward_dis --mode train &
python main.py --config ./config/reward_function/reward_dis_lya.json --id reward_dis_lya --mode train &
python main.py --config ./config/reward_function/reward_dis_no_crash.json --id reward_dis_no_crash --mode train &
python main.py --config ./config/reward_function/reward_dis_tracking.json --id reward_dis_tracking --mode train &
python main.py --config ./config/reward_function/reward_lya.json --id reward_lya --mode train &
python main.py --config ./config/reward_function/reward_lya_tracking.json --id reward_lya_tracking --mode train &
python main.py --config ./config/reward_function/reward_tracking.json --id reward_tracking --mode train