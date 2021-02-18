import torch

class cat_dataloaders():
    """Class to concatenate multiple dataloaders"""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter)) # may raise StopIteration
        return tuple(out)

class DEBUG_dataset(torch.utils.data.Dataset):
    def __init__(self,alpha):
        self.d = (torch.arange(100) + 1) * alpha
    def __len__(self):
        return self.d.shape[0]
    def __getitem__(self, index):
        return self.d[index]

train_dl1 = torch.utils.data.DataLoader(DEBUG_dataset(10), batch_size = 1,num_workers = 0 , shuffle=True)
train_dl2 = torch.utils.data.DataLoader(DEBUG_dataset(1), batch_size = 4,num_workers = 0 , shuffle=True)
tmp = cat_dataloaders([train_dl1,train_dl2])
for x in tmp:
    print(x)