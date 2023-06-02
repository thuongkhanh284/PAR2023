import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import csv
import numpy as np

class PARData(Dataset):
    def __init__(self, csv_file, transform=None, num_classes=[None, None, None, None, None]):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        targets = self.data.iloc[idx, 1:].values.astype(int)

        # One-hot encode the targets
        

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
       
        
        return image, targets[0],targets[1],targets[2],targets[3],targets[4]
        

def create_csv(file_data , folder_name , output_file , valid = False ):
    
    root_folder = '/homedir06/thtran/research/PAR2023/raw_data'
    
    fi = open(file_data)
    lines = fi.readlines()
    fi.close()
    
    output_file = os.path.join(root_folder , output_file)
    with open(output_file, 'w', encoding='UTF8' , newline='') as fo:
        writer = csv.writer(fo)
        for line in lines:
            parts = line.split(',')
            img_name = parts[0]
            img_path = os.path.join(root_folder , folder_name , img_name)
            
            if int(parts[1]) < 0:
                color_up = 0
            else:
                color_up = int(parts[1])
                
            if int(parts[2]) < 0:
                color_bottom = 0
            else:
                color_bottom = int(parts[2])
            
            gen = int(parts[3]) + 1
            bag = int(parts[4]) + 1
            hat = int(parts[5].split('\n')[0]) +  1
            
            
            rows = [img_path , str(color_up) ,  str(color_bottom) , str(gen)  , str(bag)  , str(hat) ]

            writer.writerow(rows)

def create_csv_with_only_valid(file_data , folder_name , output_file  ):
    
    root_folder = '/homedir06/thtran/research/PAR2023/raw_data'
    
    fi = open(file_data)
    lines = fi.readlines()
    fi.close()
    
    output_file = os.path.join(root_folder , output_file)
    with open(output_file, 'w', encoding='UTF8' , newline='') as fo:
        writer = csv.writer(fo)
        for line in lines:
            parts = line.split(',')
            img_name = parts[0]
            img_path = os.path.join(root_folder , folder_name , img_name)
            
            min_t = np.min(np.array([int(parts[1]) , int(parts[2]) , int(parts[3]) , int(parts[4]) , int(parts[5])]))
            if min_t >= 0:
                     
                color_up = int(parts[1]) - 1       
                color_bottom = int(parts[2]) - 1
            
                gen = int(parts[3])
                bag = int(parts[4])
                hat = int(parts[5].split('\n')[0]) 
            
            
                rows = [img_path , str(color_up) ,  str(color_bottom) , str(gen)  , str(bag)  , str(hat) ]

                writer.writerow(rows)

file_data = '/homedir06/thtran/research/PAR2023/raw_data/validation_set.txt'
folder_name = 'validation_set'
output = 'validation_set_valid.csv'

create_csv_with_only_valid(file_data , folder_name , output )
 
file_data = '/homedir06/thtran/research/PAR2023/raw_data/training_set.txt'
folder_name = 'training_set'
output = 'training_set_valid.csv'

create_csv_with_only_valid(file_data , folder_name , output )
            
            
            
    