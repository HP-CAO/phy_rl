#!/usr/bin/env sh

python ./test/test_safety_envelope_mbrl.py &
python ./test/test_safety_envelope_mbrl_no_dis.py &
python ./test/test_safety_envelope_mbrl_our.py


