#!/usr/bin/env sh


python main.py --config ./config/nips/res_our_reward_dense.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_dense_1 --mode train --force &
python main.py --config ./config/nips/res_our_reward_dense.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_dense_2 --mode train --force &
python main.py --config ./config/nips/res_our_reward_dense.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_dense_3 --mode train --force &


python main.py --config ./config/nips/res_our_reward_dense_uu.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_dense_uu_1 --mode train --force &
python main.py --config ./config/nips/res_our_reward_dense_uu.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_dense_uu_2 --mode train --force &
python main.py --config ./config/nips/res_our_reward_dense_uu.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_dense_uu_3 --mode train --force
