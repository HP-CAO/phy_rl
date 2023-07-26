#!/usr/bin/env sh


python main_a1.py --config ./config/sup/taylor_20.json --id taylor_20_1 --mode train --force &
python main_a1.py --config ./config/sup/taylor_20.json --id taylor_20_2 --mode train --force &
python main_a1.py --config ./config/sup/taylor_20.json --id taylor_20_3 --mode train --force &

python main_a1.py --config ./config/sup/taylor_25.json --id taylor_25_1 --mode train --force &
python main_a1.py --config ./config/sup/taylor_25.json --id taylor_25_2 --mode train --force &
python main_a1.py --config ./config/sup/taylor_25.json --id taylor_25_3 --mode train --force