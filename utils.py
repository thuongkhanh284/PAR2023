import yaml
import os
import pandas as pd

def process_config_files(file_yaml_path ):
    
    with open(file_yaml_path, 'r') as file:
        CFG = yaml.safe_load(file)
    return CFG
	

	
	