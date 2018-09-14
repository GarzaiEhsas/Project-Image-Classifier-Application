import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as f
from torch.autograd import variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import DeepLearningImageClassifier


def deeplearning_arg_parser():
    ''' Self note: Put some text here.'''
    
    DLapplication = argparse.ArgumentParser()

# Argument for the data directory
    DLapplication.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
# Argument for the learning rate
    DLapplication.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
# Argument for GPU
    DLapplication.add_argument('--gpu', dest="gpu", action="store", default="gpu")
# Arguemtn for save directory
    DLapplication.add_argument('--save_dir', dest="save_dir", action="store", default"./checkpoint.pth")
# Argument for epochs
    DLapplication.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
# Argument dropout
    DLapplication.add_argument('--dropout', dest="dropout", actoin="store", default=0.5)
# Argument for arch
    DLapplication.add_argument('--arch', dest="arch", action="store", default"vgg16", type=str)
#Argument for hidden units
    DLapplication.add_argument('--hidden_units', type=int, dest="hidden_units", action"store", default=120)
    return DLapplication

def checking_arguments(check_args)
    ''' Text '''
    
    # Here we are hecking if our learning rate is set correctly, else we this will set it back to 0.001
    check_args.learning_rate = 0.001 if check_args.learning_rate <= 0 else check_args.learning_rate
    #
    check_args.hidden_units = 512 if check_args.hidden_units <= 0 else check_args.hidden_units
    #
    check_args.epochs = 3 if check_args.epochs <= 0 else check_args.epochs
    
    return check_args

def main():
    parser = deeplearning_arg_parser()
    check_args = checking_arguments(check_args)
    
if __name__ == '__main__':
    main()
    
parse = DLapplication.parse_args()
path = parse.save_dir
lr = parse.learning_rate
where = parse.data_dir
structure = parse.arch
dropout = parse.dropout
hidde_layer1 = parse.hidden_units
power = parse.gpu
epochs = parse.epochs

trainloader, validationloader, testloader = DeepLearningImageClassifier.load_Data(where)

model = DeepLearningImageClassifier.model
optimizer = DeepLearningImageClassifier.optimizer
criterion = DeepLearningImageClassifier.criterion

DeepLearningImageClassifier.train_network(model, optimizer, criterion, epochs, 20, trainloader, power)

DeepLearningImageClassifier.save_checkpoint(path, structure, hidden_layer1, dropout, lr)


print("The model is now trained.")







