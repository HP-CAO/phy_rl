#!/usr/bin/env sh


python main.py --config ./config/nips_2/nips_unk_our_no_res.json --params agent_params/experience_prefill_size 5000 --id nips_unk_our_no_res --mode train --force &
python main.py --config ./config/nips_2/nips_unk_our_with_res.json --params agent_params/experience_prefill_size 5000 --id nips_unk_our_with_res --mode train --force &
python main.py --config ./config/nips_2/nips_unk_ubc_no_res.json --params agent_params/experience_prefill_size 5000 --id nips_unk_ubc_no_res --mode train --force
python main.py --config ./config/nips_2/nips_unk_ubc_with_res.json --params agent_params/experience_prefill_size 5000 --id nips_unk_ubc_with_res --mode train --force