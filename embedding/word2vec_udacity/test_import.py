import tensorflow as tf
from distutils.sysconfig import get_python_lib
print("tf version: ",tf.__version__)
print("libs:",get_python_lib())

print("compile flags:"," ".join(tf.sysconfig.get_compile_flags()))
print("link flags:"," ".join(tf.sysconfig.get_link_flags()))
