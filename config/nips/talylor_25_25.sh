#!/usr/bin/env sh


python main_a1.py --config ./config/nips/taylor_25_25.json --id taylor_25_25_1 --mode train --force &
python main_a1.py --config ./config/nips/taylor_25_25.json --id taylor_25_25_2 --mode train --force &
python main_a1.py --config ./config/nips/taylor_25_25.json --id taylor_25_25_3 --mode train --force