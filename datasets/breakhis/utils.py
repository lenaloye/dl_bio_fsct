import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
#import scanpy as sc
#from anndata import read_h5ad

def extract_and_process_csv(root, csv_name=None):
    # Read the csv file
    fold_df = pd.read_csv(os.path.join(root, csv_name))

    # Rename the column filename to path
    fold_df = fold_df.rename(columns = {"filename":"path"})

    # Create new columns for filenames, labels, and classes
    fold_df['filename'] = fold_df['path'].apply(lambda x:x.split("/")[-1])
    fold_df['label'] = fold_df['path'].apply(lambda x: x.split("/")[5])
    fold_df['class'] = pd.Categorical(fold_df['label']).codes

    return fold_df

class BHFewShotSubDataset(Dataset):
    def __init__(self, samples, category, root, dataset_name):
        self.samples = samples
        self.category = category
        self.root = root
        self._dataset_name = dataset_name
        self.transforms = transforms.Compose([transforms.Resize((224,341)),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, i):
        img_path = os.path.join(self.root, self._dataset_name, 'BreaKHis_v1', self.samples[i])
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        return img, self.category

    def __len__(self):
        return self.samples.shape[0]

    @property
    def dim(self):
        img, _ = self.__getitem__(0)
        return list(img.shape)






