#!/usr/bin/env sh

python ./test/test_safety_envelope_mbrl_our.py &
python ./test/test_safety_envelope_mbrl_ubc.py &
python ./test/test_safety_envelope_mf_our.py &
python ./test/test_safety_envelope_mf_ubc.py


