#!/usr/bin/env bash
# test io speed
#python -u eval.py --test-io=1 --batch-size=128
#
python -u eval.py --test-io=0 --batch-size=10 --gpu=2 --threshold=0.5 --visual=True
