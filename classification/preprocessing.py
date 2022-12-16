import os
import shutil
from zipfile import ZipFile
from utils.image_preprocessing import Image_preprocessing

if __name__ == "__main__":
    
    trainsets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data/trainsets/')
    
    ##### BadZipFile Error #####
    #with ZipFile(trainsets_path+'datasetA1thumbnail1.7GB.zip', 'r') as zipObj:
    #    zipObj.extractall(path=trainsets_path)
    
    #with ZipFile(trainsets_path+'datasetA2thumbnail.zip', 'r') as zipObj:
    #    zipObj.extractall(path=trainsets_path)
        
    #with ZipFile(trainsets_path+'datasetsB1B2CDE.zip', 'r') as zipObj:
    #    zipObj.extractall(path=trainsets_path)

    #with ZipFile(trainsets_path+'Virtual E Dataset.zip', 'r') as zipObj:
    #    zipObj.extractall(path=trainsets_path)
    
    shutil.copytree(trainsets_path+'datasetA2thumbnail/dataset (A2) - thumbnail', trainsets_path+'dataset (A2) - thumbnail') 
    shutil.copytree(trainsets_path+'datasetA1thumbnail1.7GB/dataset (A1) - thumbnail (1.7GB)', trainsets_path+'dataset (A1) - thumbnail (1.7GB)')
    shutil.copytree(trainsets_path+'datasetA1thumbnail1.7GB/dataset (A1) - thumbnail (1.7GB)', trainsets_path+'dataset (A1) - thumbnail (1.7GB)') 
    
    for dataset in ['B1', 'B2', 'C', 'D']:
        shutil.copytree(trainsets_path+f'datasetsB1B2CDE/datasets (B1, B2, C, D, E)/{dataset}', trainsets_path+f'dataset {dataset}')    
    
    preprocess = Image_preprocessing()
    preprocess('dataset (A1) - thumbnail (1.7GB)', 'dataset (A1) - single nails')
    preprocess('dataset (A2) - thumbnail', 'dataset (A2) - single nails', A2=True)