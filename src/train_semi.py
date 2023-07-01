import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import MultiTaskResNet
from data import PARData,create_csv_with_invalid
from PIL import Image, ImageFile
import os
from test import test_model,eval_to_modify_csv
import argparse
import numpy as np
from utils import process_config_files
ImageFile.LOAD_TRUNCATED_IMAGES = True
#  source ./anaconda3/bin/activate py37
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
parser = argparse.ArgumentParser(description='Process the config file (YAML) ')
parser.add_argument('--file', help='the yaml path to the config file',default = './Configs/config_semi.yaml')
args = parser.parse_args()

CFG = process_config_files(args.file)

OUTPUT_PTH_FOLDER = CFG['output_model']
if os.path.exists(OUTPUT_PTH_FOLDER) == False:
    os.mkdir(OUTPUT_PTH_FOLDER)
model_name = 'baseline_model_'
device = torch.device("cuda")
STOP_TRAIN = int(CFG['stop_train'])


transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),transforms.Normalize(np.array([0.485, 0.456, 0.406]),np.array([0.485, 0.456, 0.406]))])


    

def get_train_loader(  csv_file ,  batch_size , transform_t = None ):
    num_classes = [11, 11, 1, 1, 1] 
    dataset = PARData(csv_file , transform=transform_t, num_classes=num_classes)
    train_loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    return train_loader



def train_model(model , train_loader   , criterion1 ,  criterion2 , optimizer ):

    ## get train_loader 
    num_epochs = int(CFG['num_epochs'])
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        
        for inputs, targets1, targets2, targets3, targets4, targets5 in train_loader:
            i = i + 1
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)
            targets4 = targets4.to(device)
            targets5 = targets5.to(device)
            
            targets3 = targets3.unsqueeze(1)
            targets3 = targets3.float()
            targets4 = targets4.unsqueeze(1)
            targets4 = targets4.float()
            targets5 = targets5.unsqueeze(1)
            targets5 = targets5.float()
            
            outputs1, outputs2, outputs3, outputs4, outputs5 = model(inputs)

            loss1 = criterion1(outputs1, targets1)
            loss2 = criterion1(outputs2, targets2)
            loss3 = criterion2(outputs3, targets3)
            loss4 = criterion2(outputs4, targets4)
            loss5 = criterion2(outputs5, targets5)
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            optimizer.step()
            
            
            
            running_loss += loss.item()
            if i % 100 == 1:    # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i :5d}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
                
            if i==STOP_TRAIN:
                break
            
    return model

def train_semi_model():

    model = MultiTaskResNet()
    
   
    model.to(device)
    model.train()
    #criterion1 = nn.MSELoss()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= float(CFG['learning_rate']), momentum=0.9)
    
    mA_max = -1.0
    ep_semi = 0
    while (ep_semi < int(CFG['num_epochs_semi'])):
        # get
        flag = 1
        train_loader = get_train_loader( CFG['training_csv_file'] , CFG['batch_size'] , transform )
        invalid_file = CFG['invalid_csv_file']
        train_model(model , train_loader   , criterion1 ,  criterion2 , optimizer )
        mA = eval_to_modify_csv(model , invalid_file , flag , STOP_TRAIN)
        
        flag = 0
        train_loader = get_train_loader( CFG['invalid_csv_file'] , CFG['batch_size'] , transform )
        training_file = CFG['training_csv_file']
        train_model(model , train_loader   , criterion1 ,  criterion2 , optimizer)
        mA_valid = eval_to_modify_csv(model , training_file , flag , STOP_TRAIN)
        
        flag = 0
        testing_file = CFG['testing_csv_file']
        mA_test = eval_to_modify_csv(model , testing_file , flag , STOP_TRAIN)
        
        
        
        print(f" Epoch {ep_semi + 1}: mA_valid =  {mA_valid:.2f}")
        print(f" Epoch {ep_semi + 1}: mA_test =  {mA_test:.2f}")
        if mA_test + mA_valid > mA_max:
            mA_max = mA_test + mA_valid
            save_path = os.path.join(OUTPUT_PTH_FOLDER , model_name + 'ep' + str(ep_semi+1) + '_mA' + str(mA_max) + '.pth' )
            torch.save(model.state_dict(), save_path)
            
        ep_semi = ep_semi + 1
        
        # reupdate invalid file
        file_data = '/homedir06/thtran/research/PAR2023/raw_data/training_set.txt'
        folder_name = 'training_set'
        output = 'training_set_invalid.csv'
        
        create_csv_with_invalid(file_data , folder_name , output )
        print('Done updated Invalid file with -1 values ')

        
        
def main():

    train_semi_model()
    
    
    
if __name__ == "__main__":
    main()