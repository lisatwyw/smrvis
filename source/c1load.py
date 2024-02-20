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
    

def calc_tp( y_true, y_pred ):
    if isinstance( y_true, list ):
        y_pred = y_pred[0]
        y_true = y_true[0]
    A= (y_true == 1).numpy()
    B= (y_pred < .5).numpy() # negative
    n=(np.sum( A*B ) / (np.sum(A) +1e-10))
    return (- n ) # negate for when use as a loss

def calc_tn( y_true, y_pred ):
    if isinstance( y_true, list ):
        y_pred = y_pred[0]
        y_true = y_true[0]
    A= (y_true == 1).numpy()
    B= (y_pred >.5).numpy() # pos
    n=(np.sum( A*B ) / (np.sum(A) +1e-10))
    return (- n ) # negate for when use as a loss
def inverse_tpr():
    def loss(y_true, y_pred):
        r = calc_tp(y_true, y_pred)
        print('iTP', r, flush=True )
        return r
    loss.__name__ = 'inverse_tp'
    return loss
def inverse_tnr():
    def loss(y_true, y_pred):
        r=calc_tn(y_true, y_pred)
        print('iTN', r, flush=True )
        return r
    loss.__name__ = 'inverse_tn'
    return loss
 


# ---------------------- read in US volumes
if ( 'vols_trn' in  globals())==False:
    vols_trn, vols_extra = {},{}

    # --------------------- mesh
    def read_vols(e):
        for i in [1,2,3,4,5,]: # deleted the extra volume
            file='/project/def-lisat/cvprworkshop23/train_data/training/volumes/scan_%03d.mhd' % i
            h = sitk.ReadImage( file )
            vols_trn[i]=sitk.GetArrayFromImage( h )
            print('read ', file )
    read_vols(id_ex := 11)

    id_tst = IT; id_val = IV;

trn_ids = np.setdiff1d( [1,2,3,4,5], [IT, IV])

# --------------------- mesh
if (( 'pointclouds' in globals()) == False):
    plydata, verts, faces,pointclouds ={},{},{},{}
    for i in range(1,6):
        file = '/project/def-lisat/cvprworkshop23/train_data/training/meshes/scan_%03d.ply' %i
        plydata[i] = PlyData.read(file)
        vx = plydata[i]['vertex']['x']
        vy = plydata[i]['vertex']['y']
        vz = plydata[i]['vertex']['z']
        verts[i] = [ (vx[d],vy[d],vz[d]) for d in range(len(vx)) ]
        num_faces= plydata[i]['face'].count
        faces[i] = [ plydata[i]['face'][d][0] for d in range(num_faces) ]
        npts=len(verts[i])
        pointclouds[i] = us2mesh_utils.PointSampler(npts*upfac)((verts[i], faces[i]))
        print('Read', file )
    del plydata

def smart_erode(arr, m):
    '''
    https://stackoverflow.com/questions/73027495/faster-way-to-erode-dilate-images
    '''
    n = arr.shape[0]
    sd = SortedDict()
    for new in arr[:m]:
        if new in sd:
            sd[new] += 1
        else:
            sd[new] = 1
    for to_remove,new in zip(arr[:-m+1],arr[m:]):
        yield sd.keys()[0]
        if new in sd:
            sd[new] += 1
        else:
            sd[new] = 1
        if sd[to_remove] > 1:
            sd[to_remove] -= 1
        else:
            sd.pop(to_remove)
    yield sd.keys()[0]




# ---------------------- gen masks of the meshes
if ( 'masks' in globals())==False:
    sois,masks= {},{}

    # import scipy; diamond = scipy.ndimage.generate_binary_structure(rank=3, connectivity=2)

    for i in range(1,6):
        masks[i] = np.zeros( vol_sz  )

        vx=pointclouds[i][:,0]
        vy=pointclouds[i][:,1]
        vz=pointclouds[i][:,2]
        for ii,_ in tqdm(enumerate(vx)):
            px,py,pz =(vz[ii]*fac[2]).astype(int),(vy[ii]*fac[0]).astype(int),(vx[ii]*fac[1]).astype(int)
            masks[i][px,py,pz]=1

            # encode neighbours
            for r,s,t in ( (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,1) ):
                masks[i][px+r,py+s,pz+t]=1

        sois[i] = np.where( np.sum( np.sum( masks[i]>0,2),1 ) > 0 )[0]

        if EN==0:
            for ii,_ in tqdm(enumerate(vx)):
                px,py,pz =(vz[ii]*fac[2]).astype(int),(vy[ii]*fac[0]).astype(int),(vx[ii]*fac[1]).astype(int)
                masks[i][px+r*2,py+s*2,pz+t*2]=.75
                masks[i][px+r*3,py+s*3,pz+t*3]=.5
                masks[i][px+r*4,py+s*4,pz+t*4]=.25
            print( 'Dilated mask of volume', i, '; mesh vol has', masks[i].sum(), 'voxels' )
 
        def proc_masks(out1):
            if EN>1:
                out1=masks[i]*255
                out2= scipy.ndimage.gaussian_filter1d( out1, axis=1, truncate=2, sigma=EN )
                out = scipy.ndimage.gaussian_filter1d( out2*255, axis=2, truncate=2, sigma=EN )
                out = out/out.max()
                out *=255
                print( 'Before and after Gaussian blur:', out1.max(), out2.max(), out.max(),  )
            elif EN==1:
                import edt
                out = edt.edt( out1, black_border=True, parallel=4 )
                if 0:
                    print( 'Applying DT to mask')
                    out2 = scipy.ndimage.distance_transform_edt( out1)**2
                    out2 = out2/out2.max()
                    out = np.abs(out2.max() - out2)*255
                    #out = np.int8(out2)
                    print('DT\s max value:',out.max())
            return out
if EN>0:
    import FastGeodis, torch
    device = "cpu" # if torch.cuda.is_available() else "cpu"

    import time
    for i in range(1,6):
        #r=sois[i]
        tic = time.time()
        image_pt = torch.from_numpy( masks[i].astype(np.float32) ).unsqueeze_(0).unsqueeze_(0)
        image_pt = image_pt.to(device)
        mask_pt = ( torch.from_numpy(1 - masks[i].astype(np.float32)).unsqueeze_(0).unsqueeze_(0))
        mask_pt = mask_pt.to(device)

        v = 1e10
        iterations = 4
        lamb = 0.0  # <-- Euclidean distance transform
        #%timeit
        edt = FastGeodis.generalised_geodesic3d( image_pt, mask_pt, voxspacing, v, lamb, iterations ).numpy()
        print( 'Fast Geodisic Transform took', time.time() - tic, 'seconds ')
        edt = edt[0,0,:,:,:]
        masks[i]=edt
        del edt

