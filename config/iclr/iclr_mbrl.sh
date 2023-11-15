#!/usr/bin/env sh


python main_ips.py --config ./config/iclr_mb.json --id iclr_ubc_mb_1 --mode train --force &
python main_ips.py --config ./config/iclr_mb.json --id iclr_ubc_mb_2 --mode train --force &
python main_ips.py --config ./config/iclr_mb.json --id iclr_ubc_mb_3 --mode train --force &
python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our_1 --mode train --force &
python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our_2 --mode train --force &
python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our_3 --mode train --force
