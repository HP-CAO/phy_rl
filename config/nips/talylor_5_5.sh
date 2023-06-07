#!/usr/bin/env sh


python main_a1.py --config ./config/nips/taylor_5_5.json --id taylor_5_5_1 --mode train --force &
python main_a1.py --config ./config/nips/taylor_5_5.json --id taylor_5_5_2 --mode train --force &
python main_a1.py --config ./config/nips/taylor_5_5.json --id taylor_5_5_3 --mode train --force