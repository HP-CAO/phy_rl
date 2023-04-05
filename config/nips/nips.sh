#!/usr/bin/env sh


python main.py --config ./config/nips/res_our_reward_ty.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_ty --mode train --force &
python main.py --config ./config/nips/res_our_reward_dense.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_dense --mode train --force &
python main.py --config ./config/nips/res_our_reward_ty_edit.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_ty_edit --mode train --force
#python main.py --config ./config/nips/res_our_reward_ty_edit.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_ty_edit_theory --mode train --force