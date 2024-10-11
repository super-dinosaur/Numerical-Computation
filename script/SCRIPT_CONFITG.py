import os.path as osp
import sys
from icecream import ic

PATH_SCRIPT = osp.dirname(osp.abspath(__file__))
PATH_PROJECT = osp.dirname(PATH_SCRIPT) 
PATH_DATA = osp.join(PATH_PROJECT, 'data')
PATH_SCRIPT = osp.join(PATH_PROJECT, 'script')
PATH_LIB = osp.join(PATH_PROJECT, 'lib')
PATH_PROMPTS = osp.join(PATH_DATA,'prompts','prompts.json')
sys.path.append(PATH_PROJECT)
