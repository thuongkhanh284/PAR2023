import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import MultiTaskResNet
from data import PARData
from PIL import Image, ImageFile
import pandas as pd
import os
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
#  source ./anaconda3/bin/activate py37
device = torch.device("cuda")
CHECK_POINT_PATH = './Training_Models/baseline_model_ep1_mA1.0.pth'
transform_test = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),transforms.Normalize(np.array([0.485, 0.456, 0.406]),np.array([0.485, 0.456, 0.406]))])

def compare_results(targets , outputs):
    A_i = np.zeros(5)
    n = len(outputs)
    for i in range(n):
	
        if int(targets[i]) == int(outputs[i]):
            A_i[i] = 1
    return A_i

def process_outputs( outputs_torch):
    outputs = []
    scores = []
    for out in outputs_torch:
        score = out.cpu().data.numpy()
        
        idx = out.cpu().data.numpy().argmax()
        outputs.append(idx)
        scores.append(score)
    #print(outputs, '========',scores)
    return outputs , scores

def test_model (model , csv_file ):
    # will be updated soon
    test_data = pd.read_csv(csv_file)
    n = test_data.shape[0]
    
    model.to(device)
    A_results = np.zeros((n,5))
    for i in range(n):
        img_path = test_data.iloc[i, 0]
        targets = test_data.iloc[i, 1:].values.astype(int)
        
        image = Image.open(img_path).convert('RGB')
        image = transform_test(image)
        
        inputs = image.to(device)
        
        outputs1, outputs2, outputs3, outputs4, outputs5 = model(inputs.unsqueeze(0))
        outputs_torch = [outputs1, outputs2, outputs3, outputs4, outputs5]
        
        outputs , scores = process_outputs( outputs_torch )

        A_i = compare_results(targets ,outputs )
        
        A_results[i,:] = A_i
    mean_A_results = np.mean(A_results, axis = 0)
    mA = np.mean(mean_A_results)
    print(mA)
    return mA
    

    
        
def main():
    
    file_test_csv = '/homedir06/thtran/research/PAR2023/raw_data/validation_set.csv'
    model = MultiTaskResNet()
    device = torch.device("cuda")
    
    model.load_state_dict(torch.load(CHECK_POINT_PATH))
    test_model(model , file_test_csv)
    
    
    
    
if __name__ == "__main__":
    main()