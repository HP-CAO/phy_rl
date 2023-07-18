#!/usr/bin/env sh


python main_a1.py --config ./config/sup/train_baseline.json --id train_baseline_1 --mode train --force &
python main_a1.py --config ./config/sup/train_baseline.json --id train_baseline_2 --mode train --force &
python main_a1.py --config ./config/sup/train_baseline.json --id train_baseline_3 --mode train --force