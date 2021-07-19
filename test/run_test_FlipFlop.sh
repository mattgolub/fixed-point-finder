#!/bin/sh

cd ../example/

rm -r logs
python run_FlipFlop.py --train_mode=1

rm -r logs
python run_FlipFlop.py --train_mode=2

rm -r logs
python run_FlipFlop.py --train_mode=3

rm -r logs

