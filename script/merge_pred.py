import json
import numpy as np
import SCRIPT_CONFITG
import os.path as osp
import glob
import argparse

from icecream import ic
from typing import Callable
from script.plot import ploter 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=float, default=0)
    parser.add_argument('--e', type=float, default=16)
    parser.add_argument('--n', type=int, default=15)
    args = parser.parse_args()

    pred = []
    for path in glob.glob(osp.join(SCRIPT_CONFITG.PATH_DATA,'part_y','*.json')):
        with open(path,'r') as f:
            pred += json.load(f)
    assert len(pred) == 16000,f"len(pred) = {len(pred)}"
    pred = np.array(pred)
    tar_func = lambda x: 5*np.sin(1*x) + 4*np.cos(3*x)
    gt = tar_func(np.linspace(args.s,args.e,16000))
    ploter.basic_plot(
        x = np.linspace(args.s,args.e,16000),
        pred = pred,
        gt = gt,
        title=f'Pow{args.n - 1}',
    )
