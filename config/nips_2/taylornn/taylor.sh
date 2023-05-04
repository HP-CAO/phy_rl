#!/usr/bin/env sh

python main.py --config ./config/nips_2/taylornn/res_no_no_10_20.json  --id res_no_no_10_20_1 --mode train --force &
python main.py --config ./config/nips_2/taylornn/res_no_no_10_20.json  --id res_no_no_10_20_2 --mode train --force &
python main.py --config ./config/nips_2/taylornn/res_no_no_10_20.json  --id res_no_no_10_20_3 --mode train --force &

python main.py --config ./config/nips_2/taylornn/res_no_no_20_6.json --id res_no_no_20_6_1 --mode train --force &
python main.py --config ./config/nips_2/taylornn/res_no_no_20_6.json --id res_no_no_20_6_2 --mode train --force &
python main.py --config ./config/nips_2/taylornn/res_no_no_20_6.json --id res_no_no_20_6_3 --mode train --force &

python main.py --config ./config/nips_2/taylornn/res_no_no_12_12.json --id res_no_no_12_12_1 --mode train --force &
python main.py --config ./config/nips_2/taylornn/res_no_no_12_12.json --id res_no_no_12_12_2 --mode train --force &
python main.py --config ./config/nips_2/taylornn/res_no_no_12_12.json --id res_no_no_12_12_3 --mode train --force &

python main.py --config ./config/nips_2/taylornn/res_taylor_5_10.json --id res_taylor_5_10_1 --mode train --force &
python main.py --config ./config/nips_2/taylornn/res_taylor_5_10.json --id res_taylor_5_10_2 --mode train --force &
python main.py --config ./config/nips_2/taylornn/res_taylor_5_10.json --id res_taylor_5_10_3 --mode train --force
