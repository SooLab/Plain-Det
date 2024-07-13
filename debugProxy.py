import os
import sys
import runpy

# os.chdir('WORKDIR')
# args = 'python test.py 4 5'
# args = 'python tools/train_net.py --config-file projects/dino/configs/dino-resnet/dino_r50_4scale_12ep.py --num-gpus 4'
args = 'python tools/train_net.py --config-file projects/dino/configs/dino-resnet/dino_r50_4scale_12ep.py --num-gpus 2' 

args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)
if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path
sys.argv.extend(args[1:])
fun(args[0], run_name='__main__')
