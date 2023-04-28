#!/usr/bin/env sh

python main.py --config ./config/nips_2/batch2/nips_unk_1_our_with_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_1_our_with_res_1 --mode train --force &
python main.py --config ./config/nips_2/batch2/nips_unk_1_our_with_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_1_our_with_res_2 --mode train --force &
python main.py --config ./config/nips_2/batch2/nips_unk_1_our_with_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_1_our_with_res_3 --mode train --force &

python main.py --config ./config/nips_2/batch2/nips_unk_11_our_no_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_11_our_no_res_1 --mode train --force &
python main.py --config ./config/nips_2/batch2/nips_unk_11_our_no_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_11_our_no_res_2 --mode train --force &
python main.py --config ./config/nips_2/batch2/nips_unk_11_our_no_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_11_our_no_res_3 --mode train --force &

python main.py --config ./config/nips_2/batch2/nips_unk_11_our_with_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_11_our_with_res_1 --mode train --force &
python main.py --config ./config/nips_2/batch2/nips_unk_11_our_with_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_11_our_with_res_2 --mode train --force &
python main.py --config ./config/nips_2/batch2/nips_unk_11_our_with_res.json --params agent_params/experience_prefill_size 5000 --id batch_2_nips_unk_11_our_with_res_3 --mode train --force