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
    

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import tensorflow as tf; print("Tensorflow", tf.__version__)
import numpy as np



# ================================ helper ================================  
def predict_mask_in_batches( inp, model, BS ):
    assert inp.shape[1] ==model.input.shape[1]    
    
    T = inp.shape[0]            
    nsections = np.ceil( T/BS ).astype(int)    
    n_to_pad = T-nsections*BS    
    M = np.floor( T/nsections ).astype(int)

    print( 'Input intensity range:', np.min( inp ), np.max( inp ), inp.shape )

    inp = np.pad(inp, ((0,1),(0,0),(0,0)), 'edge' )       
    
    sl={}
    yp_mn = np.zeros( inp.shape )
    #yp_mx = np.zeros( inp.shape )
    #yp_ig = np.zeros( inp.shape )
      
    for b in range(0,nsections):
        
        slnums= sl[b] = np.arange( b*M, M*(b+1) )                
        print( slnums[0], slnums[-1], end=' |')
        #print(end='%d..'%b,flush=True)
        
        ii = inp[slnums,]
        ii = np.stack( (inp[slnums-1,],inp[slnums,],inp[slnums+1,]), 3)  # ignore the loop around 
        
        yp = model.predict( ii, verbose=0 )     
        yp_mn[slnums, ] = np.mean( yp, -1 )                 
        
        #yp_mx[slnums, ] = np.max( yp, axis=-1 )     
        #yp_ig[slnums, ] = yp[:,:,:,1]  
    
    print( 'Model predictions'' range:', np.min(yp_mn), np.max( yp_mn ))    
    return yp_mn[:-1,]#, yp_mx[:-1,], yp_ig[:-1,],sl # take mid slice

def jaccard(invert=True):
    def loss(y_true, y_pred):                          
        P =tf.cast(y_true > .25, dtype=tf.float32 )
        N =tf.cast(y_true < .25, dtype=tf.float32 )
        yP=tf.cast(y_pred > .50, dtype=tf.float32 )
        yN=tf.cast(y_pred < .50, dtype=tf.float32 )
        tp= P*yP
        tn= N*yN        
        fp= N*yP
        fn= P*yN   
        l= np.sum(tp)/ (np.sum(tp) + np.sum(fp) + np.sum(fn) )
        if invert:
            l = 1 -l
        return tf.Variable(l)
    loss.__name__ = 'jaccard'
    return loss

# ================================ load model and weights ================================
try:
    model = tf.keras.models.load_model('../models/IT3_IV2_EN0_IR3_ARunetpp_NF.h5', \
                                      custom_objects={'BinaryFocalCrossentropy': tf.keras.losses.BinaryFocalCrossentropy(), \
                                                     'jaccard': jaccard(False)} )      
except Exception as e:
    print(e)             

print( model.weights[0][:,1,1,0] )
print( 'Above equal to [0.04552065, 0.26180163, 0.08348185]\n?\n')
print( model.weights[0][-1,-1,1] )
print( 'Above equal to [ 0.1260792 , -0.0652371 , -0.0849674 ,  0.11226883,  0.10652731,',\
        '0.12007868,  0.1658944 , -0.04231708]\n\n?')
    

# ================================ Read in user arguments ================================    
demo=0    
ctn=1
try:
    us_filename=sys.argv[ctn];   ctn+=1
    output_file=sys.argv[ctn];   ctn+=1   
    iBS=int( sys.argv[ctn] );    ctn+=1
    thres=float( sys.argv[ctn]); ctn+=1
except:
    demo=1
    us_filename = '../models/rand2.npz'
    print('No filename provided; test data will be used...')
    output_file='../models/detected_pointcloud'
    iBS=8
    thres=0.25    
    print( '\n\n\nRun in demo mode! \n\nExample usage: test.py input.mhd output_prefix 0.25 8\nPredict in batches of 8 slices and apply global threshold of 0.25 on prb mask.' )
       
    
    
print( 'Reading %s\nWill write to %s with global thres=%.4f' %( us_filename, output_file, thres ))    

if model.input.shape[2] ==256:
    IR=3
if model.input.shape[2] ==384:
    IR=2  
if 'mhd' in us_filename:
    import SimpleITK as sitk
    hd = sitk.ReadImage( us_filename ) 
    inp = sitk.GetArrayFromImage( hd )              
    voxspacing = hd.GetSpacing()
   
    # assume all volumes are saved in RAI format, which will be read as 1280 x 768 x 768   
    if inp.shape[1] == 768:      
        assert inp.shape[2] == 768
        inp=inp[:,::IR,::IR]   
        
elif 'npz' in us_filename:   
    dat = np.load( us_filename )
    inp=dat['vol']
    voxspacing = [0.49479, 0.49479, 0.3125]  
      
# expected intensity range
inp = inp/ (1e-7+ np.max(inp))

# test output stream
np.savez_compressed( output_file, x=1, y=1, z=1 )


# ================================ Cast model predictions ================================    

# predict in batches; yp is the result of taking average over 3-slices 
yp = predict_mask_in_batches( inp, model, iBS )


# ================================ Extract points & output ================================    
pz,py,px=np.where( yp > thres )                
px,py,pz=px*voxspacing[0]*IR,py*voxspacing[1]*IR,pz*voxspacing[2]  # critical!  

print( len(px), 'points will be saved to output_file', output_file )

np.savez_compressed( output_file, x=px, y=py, z=pz )

if demo:
    print( '241,061 points saved to output_file test??')
   
    # https://www.activestate.com/resources/quick-reads/how-to-list-installed-python-packages/
    import pkg_resources
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    print(installed_packages_list)
