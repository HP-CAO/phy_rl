#!/usr/bin/env sh


python main_a1.py --config ./config/nips/baseline.json --id baseline_1 --mode train --force &
python main_a1.py --config ./config/nips/baseline.json --id baseline_2 --mode train --force &
python main_a1.py --config ./config/nips/baseline.json --id baseline_3 --mode train --force