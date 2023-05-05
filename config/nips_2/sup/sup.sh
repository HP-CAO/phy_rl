#!/usr/bin/env sh

python main.py --config ./config/nips_2/sup/nips_unk_ubc_no_res_2M.json --id nips_unk_ubc_no_res_2M_1 --mode train --force &
python main.py --config ./config/nips_2/sup/nips_unk_ubc_no_res_2M.json --id nips_unk_ubc_no_res_2M_2 --mode train --force &
python main.py --config ./config/nips_2/sup/nips_unk_ubc_no_res_2M.json --id nips_unk_ubc_no_res_2M_3 --mode train --force &

python main.py --config ./config/nips_2/sup/no_no_20_6_no_res_2M.json --id no_no_20_6_no_res_2M_1 --mode train --force &
python main.py --config ./config/nips_2/sup/no_no_20_6_no_res_2M.json --id no_no_20_6_no_res_2M_2 --mode train --force &
python main.py --config ./config/nips_2/sup/no_no_20_6_no_res_2M.json --id no_no_20_6_no_res_2M_3 --mode train --force &

python main.py --config ./config/nips_2/sup/taylor_5_10_no_res_2M.json --id taylor_5_10_no_res_2M_1 --mode train --force &
python main.py --config ./config/nips_2/sup/taylor_5_10_no_res_2M.json --id taylor_5_10_no_res_2M_2 --mode train --force &
python main.py --config ./config/nips_2/sup/taylor_5_10_no_res_2M.json --id taylor_5_10_no_res_2M_3 --mode train --force











