import os
import os.path as osp
import sys
PATH_PROJECT = osp.join(osp.dirname(osp.abspath(__file__)),'..')    
PATH_LIB = osp.join(PATH_PROJECT,'lib')
sys.path.append(PATH_PROJECT)   