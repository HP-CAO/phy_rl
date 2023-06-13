#!/usr/bin/env sh


python main_a1.py --config ./config/nips/taylor_20_20.json --id taylor_20_20_1 --mode train --force &
python main_a1.py --config ./config/nips/taylor_20_20.json --id taylor_20_20_2 --mode train --force &
python main_a1.py --config ./config/nips/taylor_20_20.json --id taylor_20_20_3 --mode train --force