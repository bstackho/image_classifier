# import required files
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse
from functions import load_data, build_classifier, train_network, test_network, save_checkpoint

#parse input parameters
parser = argparse.ArgumentParser(description='Train Image Classifier.')

parser.add_argument('data_directory', action = 'store',
help = 'Enter path to data directory.')

parser.add_argument('--save_file', action = 'store',
dest = 'save_file', default = 'checkpoint.pth',
help = 'Enter file name to checkpoint filem, default is checkpoint.pth')

parser.add_argument('--arch', action = 'store',
dest = 'architecture', default = 'densenet121',
help = 'Enter neural network architecture, default is densenet121')

parser.add_argument('--learn_rate', action = 'store',
dest = 'learning_rate', type=float,  default = '.002',
help = 'Enter learning rate, default is .002')

parser.add_argument('--hidden_units', action = 'store',
dest = 'hidden_units', type=int, default = '512',
help = 'Enter number of hidden units, default is 512')

parser.add_argument('--epochs', action = 'store',
dest = 'epochs', type=int, default = '2',
help = 'Enter number of training epochs, defaults is 2 ')

parser.add_argument('--gpu', action = "store_false", default = True,
help = 'Turn GPU mode on or off, default is on.')

results = parser.parse_args()

#variable for controlling the progrram
data_dir = results.data_directory
save_file = results.save_file
arch = results.architecture
lr = results.learning_rate
hidden_units = results.hidden_units
epochs = results.epochs
gpu_mode = results.gpu

#load the data
train_data, valid_data, test_data, trainloader, validloader, testloader = load_data(data_dir)

#load the model
model = getattr(models,arch)(pretrained=True)
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

#define the classifier
input_units = model.classifier.in_features
build_classifier(model, input_units, hidden_units)

# save the network
#save_checkpoint(model, train_data, optimizer, epochs, save_file)

#train the network
model, optimizer = train_network(model, epochs, lr, trainloader, validloader, gpu_mode)

# test the network
test_network(model, epochs, lr, testloader, gpu_mode)
    
# save the network
save_checkpoint(model, train_data, optimizer, epochs, save_file)



