
import os

file='IT5_IV5_EN0_IR3_ARseunet_NF16_KS-1_NR-1_NC2_LSbfce_OPadam_LR0.0005_BS64_WT1.0_CR3_SE2_RT50_DO0.2_NZ3_ACsigmoid_ST10_SH1'

if 'ckpt' in file:
    ff=os.path.basename(file[:-5]).split('_')
else:
    ff=os.path.basename(file).split('_')
for s in ff:
    try:
        exec( '%s=%s' %( s[:2], s[2:] ) )
        print( '%s=%s' %( s[:2], s[2:] ) )
    except:
        exec( '%s=\'%s\'' %( s[:2], s[2:] ) )
        print( '%s=\'%s\'' %( s[:2], s[2:] ) )
    
if IR==3:      
    NX=NY=266 
elif IR==2:    
    NX=NY=384 
NZ=3 

try:
    exec( open('c0args.py').read() ) 
    exec( open('c1tfsetup.py' ).read())
except:
    print('Scripts for part 2 needed')
