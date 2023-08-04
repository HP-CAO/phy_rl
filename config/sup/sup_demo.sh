#!/usr/bin/env sh


python main_a1.py --config ./config/sup/demo_taylor_15.json --id demo_taylor_15_1 --mode train --force &
python main_a1.py --config ./config/sup/demo_taylor_15.json --id demo_taylor_15_2 --mode train --force &
python main_a1.py --config ./config/sup/demo_taylor_20.json --id demo_taylor_20_1 --mode train --force &
python main_a1.py --config ./config/sup/demo_taylor_20.json --id demo_taylor_20_2 --mode train --force