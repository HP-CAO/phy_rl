#!/usr/bin/env sh


python main.py --config ./config/nips/res_our_reward_ty_edit.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_ty_edit_1 --mode train --force &
python main.py --config ./config/nips/res_our_reward_ty_edit.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_ty_edit_2 --mode train --force &
python main.py --config ./config/nips/res_our_reward_ty_edit.json --params agent_params/experience_prefill_size 5000 --id res_our_reward_ty_edit_3 --mode train --force
