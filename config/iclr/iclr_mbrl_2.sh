#!/usr/bin/env sh


python main_ips.py --config ./config/iclr_mb.json --id iclr_ubc_mb_4 --mode train --force &
python main_ips.py --config ./config/iclr_mb.json --id iclr_ubc_mb_5 --mode train --force &
python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our_4 --mode train --force &
python main_ips.py --config ./config/iclr_mb_our.json --id iclr_mb_our_5 --mode train --force
