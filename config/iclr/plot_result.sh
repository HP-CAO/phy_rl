#!/usr/bin/env sh

python ./test/iclr/test_safety_envelope_mbrl_our.py &
python ./test/iclr/test_safety_envelope_mbrl_ubc.py &
python ./test/iclr/test_safety_envelope_mf_our.py &
python ./test/iclr/test_safety_envelope_mf_ubc.py


