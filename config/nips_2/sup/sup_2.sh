#!/usr/bin/env sh

python main.py --config ./config/nips_2/sup/nips_unk_ubc_no_res_1.5M.json --id nips_unk_ubc_no_res_1.5M_1 --mode train --force &
python main.py --config ./config/nips_2/sup/nips_unk_ubc_no_res_1.5M.json --id nips_unk_ubc_no_res_1.5M_2 --mode train --force &
python main.py --config ./config/nips_2/sup/nips_unk_ubc_no_res_1.5M.json --id nips_unk_ubc_no_res_1.5M_3 --mode train --force










