import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import MultiTaskResNet
from data import PARData
from PIL import Image, ImageFile
import os
from eval import test_model
import argparse
import numpy as np
from utils import process_config_files
ImageFile.LOAD_TRUNCATED_IMAGES = True
#  source ./anaconda3/bin/activate py37

parser = argparse.ArgumentParser(description='Process the config file (YAML) ')
parser.add_argument('--file', help='the yaml path to the config file',default = './Configs/config_example.yaml')
args = parser.parse_args()

CFG = process_config_files(args.file)

OUTPUT_PTH_FOLDER = CFG['output_model']
if os.path.exists(OUTPUT_PTH_FOLDER) == False:
    os.mkdir(OUTPUT_PTH_FOLDER)
model_name = 'baseline_model_'


transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),transforms.Normalize(np.array([0.485, 0.456, 0.406]),np.array([0.485, 0.456, 0.406]))])


    

def get_train_loader(  transform_t = None ):
    num_classes = [11, 11, 1, 1, 1] 
    dataset = PARData(CFG['training_csv_file'] , transform=transform_t, num_classes=num_classes)
    train_loader = DataLoader(dataset, batch_size=int(CFG['batch_size']), shuffle=True)
    return train_loader


    
def train_model(train_loader):

    ## get train_loader 
    num_epochs = int(CFG['num_epochs'])
    model = MultiTaskResNet()
    device = torch.device("cuda")
    model.to(device)
    model.train()
    #criterion1 = nn.MSELoss()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= float(CFG['learning_rate']), momentum=0.9)
    mA_max = -1.0
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
        mA = test_model(model , CFG['testing_csv_file'])
        print(f"{epoch + 1}: mA =  {mA:.2f}")
        if mA > mA_max:
        
            save_path = os.path.join(OUTPUT_PTH_FOLDER , model_name + 'ep' + str(epoch+1) + '_mA' + str(mA) + '.pth' )
            torch.save(model.state_dict(), save_path)
            mA_max = mA
        
def main():
    train_loader = get_train_loader( transform )
    train_model( train_loader)
    
if __name__ == "__main__":
    main()