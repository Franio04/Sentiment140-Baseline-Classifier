#import sys

#print(sys.path) #this is where python looks when we import stuff

import sys
print("Python:", sys.executable)#executable)

import numpy, pandas, sklearn, matplotlib, tqdm, rich
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("tqdm:", tqdm.__version__)

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("TensorFlow: not installed or failed ->", e)

try:
    import torch, torchtext
    print("PyTorch:", torch.__version__, "| torchtext:", torchtext.__version__)
except Exception as e:
    print("PyTorch/torchtext: not installed or failed ->", e)
