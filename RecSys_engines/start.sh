#!/usr/bin/env bash

rm *.out

nohup python3 $PWD/clip.py > clip.out 2>&1 &
nohup python3 $PWD/blip.py > blip.out 2>&1 &
nohup python3 $PWD/fusion.py lda_resnet partial > lda_resnet_partial.out 2>&1 &
nohup python3 $PWD/fusion.py lda_resnet total > lda_resnet_total.out 2>&1 &
