from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch

#######################################################
#               Collate fn
#######################################################

'''
class to add padding to the batches
collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class
is used on single example
'''

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[1] for item in batch]
        #pad them using pad_sequence method from pytorch.
        source = pad_sequence(source, batch_first=False, padding_value = self.pad_idx)

        #get all target indexed sentences of the batch
        target = [item[0] for item in batch]
        return torch.tensor(target), torch.tensor(source)

def pad_length(b):
    # build vectorized sequence
    v = [encode(x[1]) for x in b]
    # compute max length of a sequence in this minibatch and length sequence itself
    len_seq = list(map(len,v))
    l = max(len_seq)
    return ( # tuple of three tensors - labels, padded features, length sequence
        torch.LongTensor([t[0]-1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t),(0,l-len(t)),mode='constant',value=0) for t in v]),
        torch.tensor(len_seq)
    )

#######################################################
#            Define Dataloader Functions
#######################################################

# If we run a next(iter(data_loader)) we get an output of batch_size * (num_workers+1)
def get_train_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True): #increase num_workers according to CPU
    #get pad_idx for collate fn
    pad_idx = dataset.source_vocab.stoi['<PAD>']
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx)) #MyCollate class runs __call__ method by default
    return loader

def get_valid_loader(dataset, train_dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True):
    pad_idx = train_dataset.source_vocab.stoi['<PAD>']
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader
