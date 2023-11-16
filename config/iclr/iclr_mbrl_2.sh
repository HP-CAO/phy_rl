#!/usr/bin/env sh


python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our --mode train --force &
python main_ips.py --config ./config/iclr_mb_ubc.json --id iclr_mb_ubc --mode train --force &
python main_ips.py --config ./config/iclr_our.json --id iclr_our_mf --mode train --force &
python main_ips.py --config ./config/iclr_ubc.json --id iclr_ubc_mf --mode train --force
