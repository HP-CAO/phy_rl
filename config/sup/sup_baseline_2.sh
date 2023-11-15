#!/usr/bin/env sh

python main_a1.py --config ./config/sup/train_baseline.json --id train_baseline_4 --mode train --force &
python main_a1.py --config ./config/sup/train_baseline.json --id train_baseline_5 --mode train --force
