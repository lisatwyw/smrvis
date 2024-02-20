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
    
exec( open('c1load.py').read() )


NZ=3 # number of depth dimensions


from scipy.spatial.distance import directed_hausdorff
import tensorflow as tf
from tensorflow.keras import backend as K



def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection) / (K.sum(K.square(y_true),axis=-1) + K.sum(K.square(y_pred),axis=-1) + epsilon)
    
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

class CustomCallback( tf.keras.callbacks.Callback ):
    def __init__(self, ohe, SR, patience=0):
        super( CustomCallback, self).__init__()
    def on_epoch_end(self, epoch, logs={} ):
        SR = self.SR
        if (epoch % 5)==0:
            # ----------------- validation   -----------------
            yp = self.model.predict( self.encoded_v )


def pcl_distance_v0(symmetric=False):
    def loss(y_true, y_pred, symmetric=False):

        bs = y_true.shape[0]
        l = 0
        s = 1
        for b in range(bs):

            gx, gy=np.where(y_true[b,:,:,s])
            px, py=np.where(y_pred[b,:,:,s] > 0.3 )

            u = np.vstack((gx,gy) ).transpose()
            v = np.vstack((px,py) ).transpose()

            print( '>>',v.shape, u.shape, end=' | ' )
            if symmetric:
                d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
            else:
                '''
                d = tfg.nn.loss.hausdorff_distance.evaluate(tf.convert_to_tensor(u), tf.convert_to_tensor(v) )
                if b==0:
                    l = d
                else:
                    l+= d
                '''
                d, iu, iv = directed_hausdorff(u, v)
                print('DD=',d, iu, iv, end=',')
        return d
    return loss

def pc_dist( u, v ):
    dt = np.zeros( (u.shape[1], v.shape[1]) )
    for d in range(3):
        ux = u[d,]
        vx = v[d,]
        c1=np.broadcast_to( ux[:, np.newaxis], ( len(ux), len(vx)) )
        c2=np.broadcast_to( vx[:, np.newaxis], ( len(vx), len(ux)) ).transpose()
        dt+= (c1 - c2)**2
    print( dt.shape ,'distance')
    q=np.argmin( dt,1) # index u
    d=np.max( np.sqrt( dt[:,q] ) )
    return d

def cdist(A, B):
    """
    Source: https://github.com/danielenricocahall/Keras-Weighted-Hausdorff-Distance-Loss/blob/master/hausdorff/hausdorff.py

    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.

    For example:

    A = [[1, 2],
         [3, 4]]

    B = [[1, 2],
         [3, 4]]

    should return:

        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D

def weighted_hausdorff_distance(w, h, alpha=.5):
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2
            
        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss


class DirectedHD( tf.keras.losses.Loss ):
    def __init__(self, symmetric=False, name="DirectedHD"):
        super().__init__(name=name)
        self.symmetric=symmetric
        self.mask_mode=None

    def call(self, y_true, y_pred):
        bs = y_true.shape[0]

        b=0 #for b in range(bs):


        s1,s2,s3=np.where(y_true[b,])

        if self.mask_mode is None:
            self.mask_mode = (y_true[b,0,0,0].numpy() == 1).astype(np.uint8)
        if self.mask_mode==1:
            th= np.quantile( y_pred[b,], .45 )
            print( 'less than threshold')
            t1,t2,t3=np.where(y_pred[b,] < th )
        else:
            th= np.quantile( y_pred[b,], .65 )
            t1,t2,t3=np.where(y_pred[b,] > th )

        u = np.vstack( (s1,s2,s3) )
        v = np.vstack( (t1,t2,t3) )
        if len(t1)>0:
            if self.symmetric:
                l = max( pc_dist(u, v), pc_dist(v, u) )
            else:
                l = pc_dist(u, v)
        else:
            l=1e7

        print(th, '-> l=', l, end=', ')
        return tf.Variable(l)/(bs+1e-7)


def random_xforms( inp, out, RT=1 ):
    if RT>0:
        r=np.random.permutation(4)
        if r[0] == 0:
            for d in range(3):
                inp[d,:,:]=np.fliplr( inp[d,:,:] )
                out[d,:,:]=np.fliplr( out[d,:,:] )
        elif r[0] == 1:
            for d in range(3):
                inp[d,:,:]=np.flipud( inp[d,:,:] )
                out[d,:,:]=np.flipud( out[d,:,:] )
        elif r[0] == 2:
            for d in range(3):
                inp[d,:,:]=np.fliplr( inp[d,:,:] )
                out[d,:,:]=np.fliplr( out[d,:,:] )
                inp[d,:,:]=np.flipud( inp[d,:,:] )
                out[d,:,:]=np.flipud( out[d,:,:] )
    return inp, out

def batch_generator( trn_ids, BS=64, show_now=0, IR=4, EN=1 ):
    while True:
        rr =np.random.permutation( np.arange(len(trn_ids)) )
        id_trn = trn_ids[rr[0]]

        
        inp = vols_trn[id_trn][:,::IR,::IR]
        out = masks[id_trn][:,::IR,::IR]

        nz, nx, ny = inp.shape
        
        if SE==1:
            # pick samples based on US images
            roi = np.where( np.sum( np.sum(inp>0,2),1 ) > 0 )[0]
        else:
            # pick samples based on meshes
            roi = np.where( np.sum( np.sum(out>0,2),1 ) > 0 )[0]

        if len(roi)<BS:
            roi = np.repeat(roi, np.round( BS/len(roi) ) )

        r1 = np.random.permutation( np.arange( len(roi) ) )

        X = np.zeros( (BS, nx, ny, NZ) )
        Y = np.zeros( (BS, nx, ny, NZ) )

        worked = []
        for i in range(BS):
            sl1 = roi[r1][i]; sl0 = sl1-1; sl2 = sl1+1

            # print( '>>', sl0, sl1, sl2, inp.shape )
            in_, ou_ = random_xforms( inp[[sl0,sl1,sl2], ], out[[sl0,sl1,sl2], ], RT  )

            if (random.random()>.999) | show_now :
                plt.close('all')
                o=ou_[ 0, :, :]
                o=255*(o/(o.max()+1e-7))
                plt.imshow( o.astype(int) )
                plt.tight_layout()
                plt.savefig( '/home/lisat/scratch/batches_%d_IR%d_EN%d_sl%d_mask.png'%(id_trn, IR, EN, i ) )
                plt.close('all')
                o=in_[ 0, :, :]
                o=255*(o/(o.max()+1e-7))
                plt.imshow( o.astype(int) )
                plt.tight_layout()
                plt.savefig( '/home/lisat/scratch//batches_%d_IR%d_EN%d_sl%d_img.png'%(id_trn, IR, EN, i ) )
            try:
                X[i,]= np.swapaxes( in_, 0,2)
                Y[i,]= np.swapaxes( ou_, 0,2)
                # if show_now:
                #    print(sl0,sl1,sl2,end=',')
                worked.append( i )
            except Exception as e:
                print(e)
                i2=worked.pop()
                sl1 = roi[r1][i2]; sl0 = sl1-1; sl2 = sl1+1
                X[i2,]= np.swapaxes( in_, 0,2)
                Y[i2,]= np.swapaxes( ou_, 0,2)
        Y= Y>0
        mx = X.max()+1e-7
        yield (X/mx), (Y *1.)


yy,xx={},{}
val_gen= batch_generator( [id_val], BS=BS, show_now=0, IR=IR)
xx[1],yy[1] = val_gen.__next__()
trn_gen= batch_generator( trn_ids, BS=BS, show_now=0, IR=IR )
xx[0],yy[0] = trn_gen.__next__()
tst_gen= batch_generator( [id_tst], BS=BS, show_now=0, IR=IR )
xx[2],yy[2] = trn_gen.__next__()


if AR=='r2unet':
    KS=-1
    # NF, NR, NC
    model = unet_variants.r2_unet(NF, NZ, width=xx[0].shape[1], height=xx[0].shape[2], input_channels=NZ, rr_layers=NR, conv_layers=NC)
elif AR=='seunet':
    NR=KS=-1
    model = unet_variants.se_unet(NF, NZ, width=xx[0].shape[1], height=xx[0].shape[2], input_channels=NZ, conv_layers=NC)
elif AR == 'wnet':
    NC=NR=-1
    _,px,py,pz = xx[0].shape
    model = unet_variants.wnet(px,py,pz, NF, KS, DO)


if OP.lower() =='adam':
    optim = tf.keras.optimizers.Adam(learning_rate= LR )
elif OP.lower() =='rms':
    optim = tf.keras.optimizers.RMSprop(learning_rate= LR )
elif OP.lower() =='sgd':
    optim = tf.keras.optimizers.SGD(learning_rate= LR )
elif OP.lower() =='nadam':
    optim = tf.keras.optimizers.Nadam( learning_rate = LR )
'''
# python 3.10 or after
match OP:
    case 'adam':
        optim = Adam(learning_rate= LR )
    case 'rms':
        optim = RMSprop(learning_rate= LR )
    case 'sgd':
        optim = SGD(learning_rate= LR )
    case 'nadam':
        optim = tf.keras.optimizers.Nadam( learning_rate = LR )
'''
if CR==2:
    decay_steps = 10000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        LR,
        decay_steps,
        end_learning_rate=LR//1000,
        power=0.5)
    optim.learning_rate=learning_rate_fn
elif CR==1:
    clr = tfa.optimizers.CyclicalLearningRate( LR, 2*LR, step_size=2*steps[0], scale_fn =lambda x: 1/(2.**(x-1) ) )
    optim.learning_rate = clr


if 'mae' in LS:
    wts=[1,2,2]
    loss_fn = [tf.keras.losses.MeanAbsoluteError(), inverse_tpr(), inverse_tnr()]
elif 'mse' in LS:
    wts=[1,2,2]
    loss_fn = [tf.keras.losses.MeanSquaredError(), inverse_tpr(), inverse_tnr()]
elif 'miou' in LS:
    wts= [1]
    loss_fn=  tf.keras.metrics.MeanIoU( num_classes= 2, name='loss')
else:
    wts= [1]
    loss_fn=DirectedHD(True)

for MODE in tasks:
    if MODE ==0:
        model.compile(run_eagerly=True, loss=loss_fn, loss_weights=wts, optimizer=optim, \
                      metrics=[ DirectedHD(False) ] )



        # ====================== get prefix ==============================================================================
        def get_prefix():
            prefix = outdir + \
                'IT%d_IV%d_EN%d_IR%d_AR%s_NF%d_KS%d_NR%d_NC%d_'%(IT, IV, EN, IR, AR, NF, KS, NR, NC) + \
                'LS%s_OP%s_LR%.4f_BS%d_WT%.1f_CR%d_SE%d_RT%d_DO%.1f' %(LS, OP, LR, BS, WT, CR, SE, RT, DO )
            try:
                os.mkdir(prefix)
            except:
                pass
            return prefix
        prefix = get_prefix()
        print('\n\nResults will be written to',prefix)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            monitor='DirectedHD',
            filepath=prefix + '/ckpt',
            mode='auto', save_best_only=True, save_weights_only=True,)
        earlystop=tf.keras.callbacks.EarlyStopping(
            monitor='DirectedHD',
            min_delta=0,
            patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=False)

        callbacks =[ model_checkpoint_callback,  earlystop, tf.keras.callbacks.CSVLogger('{}_history.csv'.format(prefix), append=True)]

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(gpu, 'set true')
            except:
                pass


    elif MODE== 1:
        print( model.summary())

        #try:
        # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)         # not avail after tf2
        hist=  model.fit( trn_gen, validation_data=val_gen, initial_epoch=0, \
                         validation_steps=steps[1], callbacks=callbacks, steps_per_epoch=steps[0], \
                         use_multiprocessing=False, epochs=EP,  )# options =run_opts )
        #except Exception as e:
        #    print(e)
        hist= hist.history

    elif MODE==2:

        def plot_progress( prefix, hist ):
            plt.close('all')
            K=hist.keys()
            colors = plt.cm.jet(np.linspace(0,1,len(K)))
            for i,k in enumerate( K ):
                m=hist[k]
                if 'val' in k:
                    lnsty='-'
                else:
                    lnsty='--'
                plt.plot( m, lnsty, linewidth=.5, color= colors[i, ], label=k )
            plt.legend()
            plt.tight_layout()
            plt.savefig( prefix+'.png')
        plot_progress( prefix, hist )

    elif MODE>2:
        #from skimage.restoration import denoise_tv_chambolle, estimate_sigma

        try:
            checkpoint = tf.train.Checkpoint(model )
            checkpoint.restore( prefix + '/ckpt' ).expect_partial()
            print( 'reloaded from checkpoint')
        except:
            print('cannot reload')


def compare( IT , q=.5):
    x = vols_trn[IT][:,::IR,::IR]
    out = masks[IT][:,::IR,::IR]

    x = np.moveaxis( np.reshape( np.swapaxes(x, 0,2)[:,:,:1278], (256,256, NZ,426) ), 3, 0)
    yp = model.predict( x )

    yp2 = np.moveaxis( yp, 0, 3 )
    yp2 = np.reshape(  yp2, (256,256, NZ*426) )
    yp2 = np.swapaxes( yp2, 0, -1 )

    th= np.quantile( yp2, q )
    res = yp2> th
    px,py,pz = np.where( res )
    print( 'Detected interest points dominate %.2f of volume' % (np.sum(res)/np.prod( yp2.shape ) ), 'n=', len(px), 'thres=', th )

    vx=pointclouds[IT][:,0]
    vy=pointclouds[IT][:,1]
    vz=pointclouds[IT][:,2]
    d = np.nan
    m = len(px)
    if (m>0) & (m<30000000):# 27,918,336
        rr=300000
        d=pc_dist( np.vstack( (px[::rr],py[::rr],pz[::rr])  ), pointclouds[IT].transpose() )
        print(d)
    return px,py,pz,vx,vy,vz,d

if MODE>0:
    try:
        px0,py0,pz0,vx0,vy0,vz0, d1 = compare( trn_ids[0], .65 )
    except Exception as e:
        print('trn',e)
    try:
        px1,py1,pz1,vx1,vy1,vz1, d2 = compare( IV, .65 )
    except Exception as e:
        print('val',e)
    try:
        px2,py2,pz2,vx2,vy2,vz2, d3 = compare( IT, .65 )
        #pd.DataFrame( [px0,py0,pz0,px1,py1,pz1,px2,py2,pz2], index=['x1','y1','z1','x2','y2','z2','x3','y3','z3',]).to_csv( prefix + '_pts.csv' )
    except Exception as e:
        print('test',e)
    pd.DataFrame( [d1,d2,d3], columns=[os.path.basename(prefix)], index=['trn0','val','test']).to_csv( prefix + '_metrics.csv' )


# AR='wnet'; LS='mae'; NF=8; KS =5; NR=2; NC=3; BS=16; tk=1; exec( open('c2tf.py', encoding='UTF-8').read() )
# AR='seunet'; LS='mae';BS=16; tk=1; exec( open('c2tf.py', encoding='UTF-8').read() )
# AR='r2unet';  LS='mae'; BS=16; tk=1; exec( open('c2tf.py', encoding='UTF-8').read() )

# initially, distances are 1120
for se in [2]:#range(3):
    print( 'casting predictions...')
    yp=model.predict(xx[se])
    print('predictions made.')
    for s in range( BS ):
        print( end='.' ,flush=True)
        plt.close('all')
        fig,ax = plt.subplots(3,3,figsize=(12,9))
        for a in range(3):
            aax=ax[0,a].imshow( yp[s,:,:,a] );  fig.colorbar(aax, ax=ax[0,a]) # ax[0,a].set_title( 'Prb channel#%d' %a );
            th=np.quantile( yp[s,:,:,a],.45)
            # sig = estimate_sigma( yp[s,:,:, a] )
            # yyp = denoise_tv_chambolle( yp[s, :,:,a], weight=.1 )
            aax=ax[1,a].imshow( yp[s,:,:,a]> th ); ax[1,a].set_title( 'Thresholded ' ); fig.colorbar(aax, ax=ax[1, a])
            aax=ax[2,a].imshow( yy[se][s,:,:,a].astype(np.uint8) ); ax[2,a].set_title( 'Groundtruth' );  fig.colorbar(aax, ax=ax[2,a])
        plt.tight_layout()
        ax[2,1].set_xlabel ( 'Result of %s when Val/Tst=%d/%d'%(tids[se], IV,IT) )
        plt.suptitle(os.path.basename( prefix) )
        plt.savefig( '%s/yp%d_%d_mode%d.png'%( prefix, se, s, MODE ) )
        print(end='>',flush=True)




