import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

# function to load data
def load_data(data_dir):

    # setup directories based off of data directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, valid_data, test_data, trainloader, validloader, testloader   

# Function to build new classifier
def build_classifier(model, input_units, hidden_units):
    
    from collections import OrderedDict
    
    # Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False  
    
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_units, hidden_units)),
    ('relu', nn.ReLU()), 
    ('fc2', nn.Linear(hidden_units, 102)),
    ('drop1', nn.Dropout(p=0.2)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    return model

def train_network(model, num_epochs, learning_rate, trainloader, validloader, gpu_mode):

    #train and validate the network
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #move model to CUDA if availavle.
    if gpu_mode: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else :
        device="cpu"
    model.to(device)

    #setup paramaters for training
    epochs = num_epochs
    print_every = 5
    steps = 0

    # train the network with the training daeta
    for e in range(epochs):
        model.train()
        running_loss = 0
        accuracy_train = 0
    
        # pull out the images and labels
        for images, labels in iter(trainloader):
            steps += 1
        
            # Move input and label tensors to CUDA if avalable.
            inputs, labels  = Variable(images), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
        
            # train the model with a forward and backward pass
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            # calculate some statistics for later (loss and accuracy)
            running_loss += loss.item()
            ps_train = torch.exp(output).data
            equality_train = (labels.data == ps_train.max(1)[1])
            accuracy_train += equality_train.type_as(torch.FloatTensor()).mean()
        
            # check every 5 steps.
            if steps % print_every == 0:
                model.eval()           
                accuracy_valid = 0
                valid_loss = 0
            
                # get imaes, lables for validation run
                for images, labels in validloader:
                    with torch.no_grad():
                    
                        # move to CUDA if available.
                        inputs, labels  = Variable(images), Variable(labels)
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                        # run the modle
                        output = model.forward(inputs)

                        # calcuation some statistics
                        valid_loss += criterion(output, labels).item()
                        ps_valid = torch.exp(output).data
                        equality_valid = (labels.data == ps_valid.max(1)[1])
                        accuracy_valid += equality_valid.type_as(torch.FloatTensor()).mean()
            
                # print some statistics
                print("Epoch: {}/{}.. ".format(e+1, epochs), 
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}..".format(valid_loss/len(validloader)),
                  "Training Accuracy: {:.3f}".format(accuracy_train/len(trainloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy_valid/len(validloader)))
            
                running_loss = 0
                model.train()
            
    return model, optimizer
    
def test_network(model, num_epochs, learning_rate, testloader, gpu_mode):
        
    # Define criterion and optimizer
    model.eval()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #move model to CUDA if availavle.
    if gpu_mode : 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else :
        device="cpu"
    model.to(device);

    #setup paramaters for testing
    accuracy_test = 0
    test_loss = 0
            
    # get imaes, lables for test run
    for images, labels in testloader:
        with torch.no_grad():
                    
            # move to CUDA if available.
            inputs, labels  = Variable(images), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
                    
            # run the model
            output = model.forward(inputs)

            # calcuate some statistics
            test_loss += criterion(output, labels).item()
            ps_test = torch.exp(output).data
            equality_test = (labels.data == ps_test.max(1)[1])
            accuracy_test += equality_test.type_as(torch.FloatTensor()).mean()
                    
            # print some statistics
            print("Test Loss: {:.3f}..".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy_test/len(testloader)))
    return

def save_checkpoint(model, train_data, optimizer, epochs, save_file):
    #checkpoint the model and some variables.                
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'state_dict' : model.state_dict(),
                  'classifier' : model.classifier,
                  'class_to_idx' : model.class_to_idx,
                  'optimizer_state_dict' : optimizer.state_dict,
                  'epochs': epochs
                 }
    
    print("Model saved to: ", save_file)
                    
    return torch.save(checkpoint, save_file)


def load_checkpoint(model, save_dir, gpu_mode):
    
    
    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
    else:
        checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def load_checkpoint2(model, save_file):
    
    #load the checkpoint
    
    checkpoint = torch.load(save_file)
    
    #load the attributes of the model  
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer.state_dict = checkpoint['optimizer_state_dict']
    model.epochs = checkpoint['epochs']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Converting image to PIL image using image file path

    from PIL import Image
    pil_image = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    ## Transforming image for use with network
    pil_tfd = transform(pil_image)
    
    return pil_tfd

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    import numpy as np
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image_data = process_image(image_path)
    
    # No need for GPU on this part (just causes problems)
    model.to("cpu")
    
    # Set model to evaluate      
    model_predict = model.eval()
    
    # got to manipulate the input imnage
    inputs = Variable(image_data.unsqueeze(0))

    # run the model and calucalte probabilty
    output = model_predict(inputs)
    ps = torch.exp(output).data
    
    # calculate the top_k probabilities
    ps_top = ps.topk(top_k)
    
    # handle some model to class indexing issues
    #class_to_idx = train_data.class_to_idx
    class_to_idx = model.class_to_idx
    model.idx_to_class = inv_map = {v: k for k, v in class_to_idx.items()}
    idx2class = model.idx_to_class
    
    # calculate probs and clases
    probs = ps_top[0].tolist()[0]
    classes = [idx2class[i] for i in ps_top[1].tolist()[0]]
    
    return probs, classes


