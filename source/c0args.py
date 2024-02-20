#    Copyright 2023 lisatwyw Lisa Y.W. Tang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
    
SEED =101
import random, os, sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure, morphology, restoration, segmentation, transform, util)
#import napari
from plyfile import PlyData, PlyElement
from matplotlib import pyplot as plt

import pandas as pd
import tensorflow as tf
print( '\n\nGPU???', tf.config.list_physical_devices('GPU') )


# --------------- mesh processing


from importlib import reload # faciliate rapid prototyping
import us2mesh_utils, unet_variants
reload(us2mesh_utils)
reload(unet_variants)


from plyfile import PlyData, PlyElement
from matplotlib import pyplot as plt
#from mesh_to_sdf import *
#import trimesh

import SimpleITK as sitk
#import itk

from tqdm import tqdm
import skimage
from scipy import signal
from scipy import misc

def fix_all_seeds( SEED ):
    np.random.seed( SEED )
    try:
        torch.manual_seed( SEED )
        torch.cuda.manual_seed( SEED )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    except Exception as e:
        print(e)
    try:
        tf.random.set_seed( SEED )
    except:
        pass
    os.environ["PYTHONHASHSEED"] = str( SEED )
fix_all_seeds( SEED )







# ---------------------- constants
vol_sz = [1280, 768, 768]

voxspacing = [0.49479, 0.49479, 0.3125]
fac = 1/np.asarray(voxspacing)

upfac=2


outdir = '/home/lisat/scratch/0522/' #'/project/def-lisat/cvprworkshop23/out/0519/'
steps  = [1,1] # steps per epoch; also used in CR
try:
    os.mkdir(outdir)
except:
    pass

print('\nInput arguments:')
for i,k in enumerate(sys.argv):
    print(i, k)
try:
    # python e1run.py $OU 0 5 $BS $VL $CL $NR  $EE $AR $NL $NU $L2 $BN $DO $LS $LR $TU $WT $CR $EX $OR
    ctn=1
    OU= sys.argv[ctn]; ctn+=1; print('>',ctn)
except:
    pass


if ( 'EN' in globals())==False:
    EN = 0
if ( 'AR' in globals())==False:
    AR = 'wnet'
    NF,KS,NC=8,5,3

if (AR == 'seunet') & (( 'NF' in globals() )==False):
    NF = 8

if 'res' in AR:
    if ( 'FT' in globals())==False:
        FT = 0
        print('\n\n\n\nBS set to!!!',BS)
        LS = 'diceloss'

if ( 'NR' in globals())==False:
    NF, NR, NC = 16,2,2

if ( 'EP' in globals())==False:
    EP = 300
if ( 'BS' in globals())==False:
    BS = 16
if ( 'OP' in globals())==False:
    OP = 'adam'
if ( 'IR' in globals())==False:
    IR =3
if ( 'LR' in globals())==False:
    LR =0.008
if ( 'DO' in globals())==False:
    DO =0.2
if ( 'CR' in globals())==False:
    CR = 1

if ( 'RT' in globals())==False:
    RT = 1 # random tforms ?


if ( 'tk' in globals())==False:
    tk =0  # running mode

if ( 'IT' in globals())==False:
    IT = 1  # test US
if ( 'IV' in globals())==False:
    IV = 2  # validation US
if ( 'SE' in globals())==False:
    SE = 2

if ( 'LS' in globals())==False:
    LS = 'pdist'; #'miou'


nlosses= len(LS.split('+'))
if nlosses==1:
    WT = 1
elif ( 'WT' in globals())==False:
    WT =[ 1/nlosses]* nlosses


tids =['trn','val','tst']
demo = 0
if tk==0:
    demo = 1

if tk==1:
    if 'multi' in LS:
        tasks = [0,1,2,]
    else:
        tasks = [0,1,2,]
elif tk==3:
    tasks=[0,3]
