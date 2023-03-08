#!/usr/bin/env sh

python main.py --config ./config/cdc/no_res_ubc_reward.json --id no_res_ubc_reward_2 --mode train --force &
python main.py --config ./config/cdc/res_our_reward.json --id res_our_reward_2 --mode train --force &
python main.py --config ./config/cdc/res_ubc_reward.json --id res_ubc_reward_2 --mode train --force &

python main.py --config ./config/cdc/no_res_ubc_reward.json --id no_res_ubc_reward_3 --mode train --force &
python main.py --config ./config/cdc/res_our_reward.json --id res_our_reward_3 --mode train --force &
python main.py --config ./config/cdc/res_ubc_reward.json --id res_ubc_reward_3 --mode train --force
