import os.path as osp
from icecream import ic

PATH_PROJECT = osp.dirname(osp.abspath(__file__))
PATH_DATA = osp.join(PATH_PROJECT, 'data')
PATH_SCRIPT = osp.join(PATH_PROJECT, 'script')
PATH_LIB = osp.join(PATH_PROJECT, 'lib')