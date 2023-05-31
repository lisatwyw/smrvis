import platform; print(platform.platform())
import sys; print("Python", sys.version)
import tensorflow as tf; print("Tensorflow", tf.__version__)
import numpy as np

# https://www.activestate.com/resources/quick-reads/how-to-list-installed-python-packages/
import pkg_resources
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
   for i in installed_packages])

print(installed_packages_list)
          
def predict_mask_in_batches( inp, model, BS ):        
    assert inp.shape[1] ==model.input.shape[1]    
    
    T = inp.shape[0]            
    nsections = np.ceil( T/BS ).astype(int)    
    n_to_pad = T-nsections*BS    
    M = np.floor( T/nsections ).astype(int)

    print( 'Input intensity range:', np.min( inp ), np.max( inp ) )

    inp = np.pad(inp, ((0,1),(0,0),(0,0)), 'edge' )       
    
    sl={}
    yp_mn = np.zeros( inp.shape )
    yp_mx = np.zeros( inp.shape )
    yp_ig = np.zeros( inp.shape )
      
    for b in range(0,nsections):
        print(end='.',flush=True)
        sl[b]=slnums= np.arange( b*M, M*(b+1) )                
        ii = inp[slnums,]
        ii = np.stack( (inp[slnums-1,],inp[slnums,],inp[slnums+1,]), 3)  # ignore the loop around 
        
        yp = model.predict( ii, verbose=0 )     
        yp_mn[slnums, ] = np.mean( yp, -1 ) 
        yp_mx[slnums, ] = np.max( yp, axis=-1 )     
        yp_ig[slnums, ] = yp[:,:,:,1]  
    
    print( 'Model predictions'' range:', yp_mn.min(), yp_mn.max() )    
    return yp_mn[:-1,], yp_mx[:-1,], yp_ig[:-1,] # take mid slice

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

try:
   model = tf.keras.models.load_model('../models/IT3_IV2_EN0_IR3_ARunetpp_NF.h5', \
                                      custom_objects={'BinaryFocalCrossentropy': tf.keras.losses.BinaryFocalCrossentropy(), \
                                                     'jaccard': jaccard(False)} )      
except Exception as e:
   print(e)          
   
print( model.weights[0][:,1,1,0] ,'\n\n')
print( 'model.weights[0][:,1,1,0]\n<tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.04552065, 0.26180163, 0.08348185], dtype=float32)>')
                                                                                                
ctn=1
try:
   us_filename=sys.argv[ctn]; ++ctn
except:
   us_filename = 'rand2.npz'
   print('No filename provided; test data will be used...')
try:
   output_file=sys.argv[ctn];++ctn
except:
   output_file='test_output'
try:
   thres=float( sys.argv[ctn] ); 
except:
   thres=0.25
   
if 'mhd' in us_filename:
   import SimpleITK as sitk
   hd = sitk.ReadImage( us_filename ) 
   inp = sitk.GetArrayFromImage( hd )              
   voxspacing = hd.GetSpacing()

elif 'npz' in us_filename:   
   dat = np.load( us_filename )
   inp=dat['vol']
   voxspacing = [0.49479, 0.49479, 0.3125]
   
assert inp.shape[1] == 768
assert inp.shape[2] == 768

# expected intensity range
inp = inp/ (1e-7+ np.max(inp))

# predict in batches
yp,_,_ = predict_mask_in_batches( inp, model, 16 )

if model.input.shape[2] ==256:
   IR=3
if model.input.shape[2] ==384:
   IR=2
   
pz,py,px=np.where( yp > thres )                
px,py,pz=px*voxspacing[0]*IR,py*voxspacing[1]*IR,pz*voxspacing[2]  # critical!  

print( len(px), 'points will be saved to output_file', output_file )

np.savez_compressed( output_file, x=px, y=py, z=pz )
      
   
