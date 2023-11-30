import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.breakhis.utils import *
from datasets.dataset import *
from abc import ABC
from PIL import Image


class BHDataset(FewShotDataset, ABC):
    _dataset_name = 'breakhis'
    _dataset_url = 'https://www.kaggle.com/datasets/ambarish/breakhis'

    def load_breakhis(self, mode='train'):
        # Extract the data
        df = extract_and_process_csv(root=self._data_dir, csv_name='Folds.csv')
        
        # Differentiate and select the wanted types
        train_type = ['lobular_carcinoma', 'mucinous_carcinoma', 'fibroadenoma', 'ductal_carcinoma']
        val_type = ['phyllodes_tumor','papillary_carcinoma']
        test_type = ['adenosis', 'tubular_adenoma']
        split = {'train': train_type,
                 'val': val_type,
                 'test': test_type}
        
        types = split[mode]
        
        # Subset data based on target types
        df = df[df['label'].isin(types)]
        df = df.reset_index(drop=True)
        
        # Convert path to torch tensor x
        samples = df['path'].to_numpy(dtype=str)

        # Convert class to torch tensor y
        targets = df['class'].to_numpy(dtype=np.int32)
        
        return samples, targets

class BHSimpleDataset(BHDataset):
    
    def __init__(self, batch_size, root='./data/', mode='train'):
        self.initialize_data_dir(root, download_flag=False)
        self.samples, self.targets = self.load_breakhis(mode)
        self.batch_size = batch_size
        self.root = root
        super().__init__()
        self.transforms = transforms.Compose([transforms.Resize((224,341)),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, i):
        img_path, label = os.path.join(self.root, self._dataset_name, 'BreaKHis_v1', self.samples[i]), int(self.targets[i])
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return self.samples.shape[0]

    @property
    def dim(self):
        img, _ = self.__getitem__(0)
        return list(img.shape)

    def get_data_loader(self) -> DataLoader:
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)

        return data_loader


class BHSetDataset(BHDataset):

    def __init__(self, n_way, n_support, n_query, n_episode=100, root='./data', mode='train'):
        self.initialize_data_dir(root, download_flag=False)

        self.n_way = n_way
        self.n_episode = n_episode
        min_samples = n_support + n_query

        self.root = root
        
        samples_all, targets_all = self.load_breakhis(mode)
        self.categories = np.unique(targets_all)  # Unique types labels

        self.transforms = transforms.Compose([transforms.Resize((224,341)),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img_path = os.path.join(self.root, self._dataset_name, 'BreaKHis_v1', samples_all[0])
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        self.x_dim = list(img.shape)

        self.sub_dataloader = []

        sub_data_loader_params = dict(batch_size=min_samples,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.categories:
            samples = samples_all[targets_all == cl]
            sub_dataset = BHFewShotSubDataset(samples, cl, self.root, self._dataset_name)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        img, _ = sub_dataset[0]

        super().__init__()

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.categories)

    @property
    def dim(self):
        return self.x_dim

    def get_data_loader(self) -> DataLoader:
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)
        return data_loader