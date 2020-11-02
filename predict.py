# import required files
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse
from functions import load_checkpoint, process_image, predict, load_data
import json
import matplotlib.pyplot as plt

# matlab stuff
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

#parse input parameters
parser = argparse.ArgumentParser(description='Predict Image.')

parser.add_argument('path_to_image', action = 'store',
help = 'Enter path to image')

parser.add_argument('save_file', action = 'store',
help = 'Enter file name to checkpoint file')

parser.add_argument('--top_k', action = 'store',
dest = 'top_k', type=int, default = 5,
help = 'Enter topK value, default is 5')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Enter path to catalog.')

parser.add_argument('--gpu', action = "store_true", default = False,
help = 'Turn GPU mode on or off, default is off.')

parser.add_argument('--arch', action = 'store',
dest = 'architecture', default = 'densenet121',
help = 'Enter neural network architecture, default is densenet121')

parser.add_argument('--data_dir', action = 'store',
dest = 'data_dir', default = 'flowers',  
help = 'Enter path to data directory.')

results = parser.parse_args()

#variable for controlling the program
image = results.path_to_image
save_file = results.save_file
top_k = results.top_k
cat_names = results.cat_name_dir
gpu_mode = results.gpu
arch = results.architecture
data_dir = results.data_dir



#load the model
model = getattr(models,arch)(pretrained=True)
load_checkpoint(model, save_file, gpu_mode)

#process image
transformed_image = process_image(image)

# predict the image
probs, classes = predict(image, model, top_k)

# get the categoriy name  from json file.
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)
class_names = []
for i in classes:
    class_names += [cat_to_name[i]]

# print top_k results to the screen
print("\n")
for i in range(0, top_k):
   print(f"This flower is predicted to be a: '{class_names[i]}' with a probability of {round(probs[i]*100,2)}% ")



    

