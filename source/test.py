import platform; print(platform.platform())
import sys; print("Python", sys.version)
import tensorflow as tf; print("Tensorflow", tf.__version__)


# https://www.activestate.com/resources/quick-reads/how-to-list-installed-python-packages/
import pkg_resources
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
   for i in installed_packages])

print(installed_packages_list)


print('Testing')


import SimpleITK as sitk
from tqdm import tqdm

model = tf.keras.models.load_model( '../models/IT3_IV2_EN0_IR3_ARunetpp_NF.h5')
print( model.summary() )

ctn =1
try:
   us_filename=sys.argv[ctn]
except:
   print('No filename provided')
