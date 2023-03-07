#!/usr/bin/env sh

python main.py --config ./config/cdc/no_res_ubc_reward.json --id no_res_ubc_reward --mode train --force &
python main.py --config ./config/cdc/res_our_reward.json --id res_our_reward --mode train --force &
python main.py --config ./config/cdc/res_ubc_reward.json --id res_ubc_reward --mode train --force
