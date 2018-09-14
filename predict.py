import matplotlib.pyplot as plt
import numpy as np
import toch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.vision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import image
import argparse

import DeepLearningImageClassifier

DLapplication = argparse.ArgumentParser(
    description='predict-file')

DLapplication.add_argument('input_img', default='paind-project/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
DLapplication.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action"store", type = str)
DLapplication.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
DLapplication.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
DLapplication.add_argument('--gpu,', default="gpu", action="store", dest="gpu")


parse = DLapplication.parse_args()
image_paths = parse.input_img
output_numbers = parse.top_k
input_img = parse.input_img
power = parse.gpu
path = parse.checkpoint

loader_trainer, loader_testing, loader_validation = DeepLearningImageClassifier.load_Data(where)

DeepLearningImageClassifier.save_checkpoint(path)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
    
probab = DeepLearningImageClassifier.predict(image_paths, model, output_numbers, power)

labeling = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probab = np.array(probab[0][0]
                  
                                   
i=0
    
while i < number of outputs:
    print("{} Probability {}".format(labels[i], probab[i]))
     i += 1
                  
print("Done!")
                  

