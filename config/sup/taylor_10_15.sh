#!/usr/bin/env sh


python main_a1.py --config ./config/sup/taylor_10.json --id taylor_10_1 --mode train --force &
python main_a1.py --config ./config/sup/taylor_10.json --id taylor_10_2 --mode train --force &
python main_a1.py --config ./config/sup/taylor_10.json --id taylor_10_3 --mode train --force &

python main_a1.py --config ./config/sup/taylor_15.json --id taylor_15_1 --mode train --force &
python main_a1.py --config ./config/sup/taylor_15.json --id taylor_15_2 --mode train --force &
python main_a1.py --config ./config/sup/taylor_15.json --id taylor_15_3 --mode train --force