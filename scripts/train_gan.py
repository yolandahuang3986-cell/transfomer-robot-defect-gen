#!/usr/bin/env python3
"""Run pinned GAN with strictly frozen training inputs; see worker.log for progress."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.generation.gan_adapter import GANDefectGenerator

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','manifest','stage','upstream','python','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--image-size',type=int,default=256)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--decay-epochs',type=int,default=1)
    p.add_argument('--max-steps',type=int,default=1,help='0 completes epochs; positive means smoke-only')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.max_steps<0:p.error('max-steps must be nonnegative')
    g=GANDefectGenerator(a.upstream,a.python,device=a.device)
    g.fit(a.stage,vars(a))
    print('GAN worker completed; see '+str(g.checkpoint.parent/'training_report.json'))
