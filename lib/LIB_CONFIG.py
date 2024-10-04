import os.path as osp
import sys
from icecream import ic

PATH_PROJECT = osp.join(osp.dirname(osp.abspath(__file__)),'..')
PATH_LIB = osp.join(PATH_PROJECT,'lib')
PATH_DATA = osp.join(PATH_PROJECT,'data')
sys.path.append(PATH_LIB)
ic(PATH_PROJECT)