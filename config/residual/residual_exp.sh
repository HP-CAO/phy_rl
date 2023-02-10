#!/usr/bin/env sh

python main.py --config ./config/residual/cartpole_ddpg.json --id cartpole_ddpg --mode train --force &
python main.py --config ./config/residual/cartpole_ddpg_res.json --id cartpole_ddpg_res --mode train --force &
python main.py --config ./config/residual/cartpole_ddpg_res_w.o_tracking_error.json --id cartpole_ddpg_res_w.o_tracking_error --mode train --force &
python main.py --config ./config/residual/cartpole_ddpg_res_distance.json --id cartpole_ddpg_res_distance --mode train --force
python main.py --config ./config/residual/cartpole_ddpg_res_sparse_reset.json --id cartpole_ddpg_res_sparse_reset --mode train --force