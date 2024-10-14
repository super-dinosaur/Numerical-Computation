#!/bin/bash
N=4
START=1
END=5
K=4
STEP=($END-$START)/$K

python script/dump_prompts.py --n $N --s 1 --e 2
# for ((i=$START; i<$END; i+=$STEP)); do
#     python script/dump_prompts.py --n $N --s $i --e $((i + $STEP))
#     python script/mul_pred.py
# done
# python script/merge_pred.py --s $START --e $END --n $((N-1))