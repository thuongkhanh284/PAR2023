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
import csv

ImageFile.LOAD_TRUNCATED_IMAGES = True
#  source ./anaconda3/bin/activate py37
device = torch.device("cuda")
#CHECK_POINT_PATH = './/Training_Models_Semi/baseline_model_ep1_mA0.0005962819275535526.pth'
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

def update_output(targets , outputs):
    A_i = np.zeros(5)
    n = len(outputs)
    for i in range(n):
    
        if int(targets[i]) == -1:
            A_i[i] = outputs[i]
        else:
            A_i[i] = targets[i]
    return A_i

def update_csv(csv_file , list_files ,list_predict):
    print(' Starting updating file  ',csv_file)
    with open(csv_file, 'w', encoding='UTF8' , newline='') as fo:
        writer = csv.writer(fo)
        
        
        n_file = len(list_files)
        for i in range(n_file):
            img_path = list_files[i]
            
            values = list_predict[i]
            
            rows = [img_path , str(int(values[0])) ,  str(int(values[1])) , str(int(values[2]))  , str(int(values[3]))  , str(int(values[4])) ]

            writer.writerow(rows)
    print(' Done updating file  ',csv_file)
    
def process_outputs( outputs_torch):
    outputs = []
    scores = []
    i = 0
    for out in outputs_torch:
        if i < 2:
            score = out.cpu().data.numpy()
            
            idx = out.cpu().data.numpy().argmax()
            
            outputs.append(idx)
            scores.append(score[0])
        else:
            score = out.cpu().data.numpy()
            idx = 0
            if score[0,0] > 0.5:
                idx = 1
            outputs.append(idx)
            scores.append(score[0,0])
            
        i = i + 1
    #print(outputs, '========',scores)
    return outputs , scores
    

def test_model (model , csv_file ):
    print('Starting testing ')
    # will be updated soon
    test_data = pd.read_csv(csv_file , header = None)
    n = test_data.shape[0]
    
    model.to(device)
    model.eval()
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
    
    return mA
    
def eval_to_modify_csv(model , csv_file , flag ,stop_train = -1):
    print('Starting Evaluation file  ',csv_file)
    test_data = pd.read_csv(csv_file)
    n = test_data.shape[0]
    
    model.to(device)
    model.eval()
    A_results = np.zeros((n,5))
    
    
    list_files = []
    list_predict = []
    if stop_train > 0:
        if n >=  stop_train:
            n = stop_train
        else:
            n = n
    
    if flag == 0:
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
        print('Done  Evaluation file  ',csv_file)
        return mA
    else:
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
            list_files.append(img_path)
            
            outputs = update_output(targets , outputs)
            list_predict.append(outputs)
            
        
        mean_A_results = np.mean(A_results, axis = 0)
        mA = np.mean(mean_A_results)
        
        update_csv(csv_file , list_files ,list_predict )
        print('Done  Evaluation file  ',csv_file)
        return mA
    
        
def main():
    
    file_test_csv = '/homedir06/thtran/research/PAR2023/raw_data/validation_set.csv'
    model = MultiTaskResNet()
    device = torch.device("cuda")
    
    model.load_state_dict(torch.load(CHECK_POINT_PATH))
    test_model(model , file_test_csv)
    
    
    
    
if __name__ == "__main__":
    main()