#!/usr/bin/env sh

#python main.py --config ./config/cdc/no_res_ubc_reward.json --id no_res_ubc_reward_cdc --mode train --force &
#python main.py --config ./config/cdc/res_our_reward.json --id res_our_reward_cdc --mode train --force &
#python main.py --config ./config/cdc/res_ubc_reward.json --id res_ubc_reward_cdc --mode train --force &
#python main.py --config ./config/cdc/res_dis_reward.json --id res_dis_reward_cdc --mode train --force &
#python main.py --config ./config/cdc/res_our_dis_reward.json --id res_our_dis_reward_cdc --mode train --force &
#python main.py --config ./config/cdc/res_ubc_dis_reward.json --id res_ubc_dis_reward_cdc --mode train --force

python main.py --config ./config/cdc/baseline.json --params agent_params/experience_prefill_size 5000 --id baseline_prefill --mode train --force &
python main.py --config ./config/cdc/no_res_ubc_reward.json --params agent_params/experience_prefill_size 5000 --id no_res_ubc_reward_cdc_prefill --mode train --force &
python main.py --config ./config/cdc/res_our_reward.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_cdc_prefill --mode train --force &
python main.py --config ./config/cdc/res_ubc_reward.json --params agent_params/experience_prefill_size 5000 --id res_ubc_reward_cdc_prefill --mode train --force &
python main.py --config ./config/cdc/res_dis_reward.json --params agent_params/experience_prefill_size 5000 --id res_dis_reward_cdc_prefill --mode train --force &
python main.py --config ./config/cdc/res_our_dis_reward.json --params agent_params/experience_prefill_size 5000 --id res_our_dis_reward_cdc_prefill --mode train --force &
python main.py --config ./config/cdc/res_ubc_dis_reward.json --params agent_params/experience_prefill_size 5000 --id res_ubc_dis_reward_cdc_prefill --mode train --force
