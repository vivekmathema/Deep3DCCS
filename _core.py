'''
_core.py
Core utilities module for Deep3DCCS framework. Provides fundamental functions including 
GPU selection, random seed management, system warnings suppression, and memory monitoring. 
Handles TensorFlow configuration and environment setup for reproducible 3DCNN operations. 
Includes logo display and timestamp utilities for consistent application branding. 
Essential foundation for molecular structure processing and CCS prediction workflows.
'''

import os
import sys
import csv
import tensorflow as tf
import datetime
from termcolor import colored
from colorama import *
import random
from PIL import Image
#==================for 3D molecule proprocessing
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem
import math as m
import numpy as np
#==============================

def show_logo():
    print(colored("\n"+30* "__"+"\n\n", "green"))

    m= "    :::::::::  :::::::::: :::::::::: :::::::::   ::::::::  :::::::::   ::::::::   ::::::::   ::::::::  "+ "\n"
    m+="    :+:    :+: :+:        :+:        :+:    :+: :+:    :+: :+:    :+: :+:    :+: :+:    :+: :+:    :+: "+ "\n"
    m+="    +:+    +:+ +:+        +:+        +:+    +:+        +:+ +:+    +:+ +:+        +:+        +:+        "+ "\n" 
    m+="    +#+    +:+ +#++:++#   +#++:++#   +#++:++#+      +#++:  +#+    +:+ +#+        +#+        +#++:++#++ "+ "\n"
    m+="    +#+    +#+ +#++:++#   +#++:++#   +#++:++#+      +#++:  +#+    +:+ +#+        +#+        +#++:++#++ "+ "\n"
    m+="    +#+    +#+ +#+        +#+        +#+               +#+ +#+    +#+ +#+        +#+               +#+ "+ "\n"
    m+="    #+#    #+# #+#        #+#        #+#        #+#    #+# #+#    #+# #+#    #+# #+#    #+# #+#    #+# "+ "\n"
    m+="    #########  ########## ########## ###         ########  #########   ########   ########   ########  "+ "\n"
    print(colored(m,"green"))
    print(colored("Deep3DCCS: A deep learnining approach for prediction of CCS using MS spectrum information", "blue"))
    print(colored("Siriraj Metabolomics & Phenomics Center (SiMPC), Mahidol University, Bangkok,Thailand", "blue"))
    print(colored("\nRunning on:[python %s]"%str(sys.version)), "blue")
    print(colored(30* "__", "green"))


def time_stamp():
    return  datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")  # Fixed: datetime.datetime.now()

def shorten_path(filepath):
    # Get the file name from the path
    filename = os.path.basename(filepath)     
    if len(filepath) > 40:   # Shorten the path to the first 5 characters, followed by ellipses and the file name for display purpose
        shortened_path = filepath[:10] + '....' + filename
    else:
        shortened_path = filepath
    return shortened_path

def set_random_seed(random_seed):
    random.seed(random_seed)                   
    np.random.seed(random_seed)               

    if tf.__version__ < "2.0":                 
        tf.set_random_seed(random_seed)
    else:
        tf.random.set_seed(random_seed)          


# Suppress TensorFlow warning messages
def hide_tf_warnings(suppress_msg=False):
    if suppress_msg:
        try:
            warnings.filterwarnings("ignore")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] =  '3'
            os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')
        except:
            pass


class ClassName(object):
    """docstring for ClassName"""
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg
        
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, (list, tuple)):
        size += sum(get_size(item, seen) for item in obj)
    elif isinstance(obj, dict):
        size += sum(get_size(v, seen) for v in obj.values())
        size += sum(get_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(item, seen) for item in obj)
    return size / (1024 * 1024)