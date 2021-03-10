import sys, copy, torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
from torch import optim

from ssl.utils.datasets import get_dataset

class PredictionModel(nn.Module):
    
    def __init__(self, encoder, pred_head, dim, out_dim):
        
        super(PredictionModel, self).__init__()
        self.encoder = encoder
        
        if pred_head is not None:
            self.pred_head = pred_head
        else:
            self.pred_head = nn.Linear(dim, out_dim)
        
    def forward(self, data):
        
        zg = self.encoder(data)
        out = self.pred_head(zg)
        
        return out
        

class EvalUnsupevised(object):
    
    def __init__(self, dataset, metric):
        
        pass
    
    def evaluate(self, learning_model, encoder):
        
        pass
    
    
    
class EvalSemisupevised(object):
    
    def __init__(self, dataset, label_rate, out_dim, 
                 task='clf', metric='acc', n_folds=10, device=None):
        
        self.dataset = get_dataset(dataset)
        self.label_rate = label_rate
        self.out_dim = out_dim
        self.task = task
        self.metric = metric
        self.n_folds = n_folds
        self.device = device
        
        # Use default config if not further specified
        self.setup_train_config()
        
        
    def setup_train_config(self, batch_size = 128,
                           p_optim = 'Adam', p_lr = 0.0001, p_weight_decay = 0, p_epoch = 100,
                           f_optim = 'Adam', f_lr = 0.001, f_weight_decay = 0, f_epoch = 100):
        
        self.batch_size = batch_size
        
        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        
    
    def evaluate(self, learning_model, encoder, pred_head=None, fold_seed=12345):
        '''
        Args:
            learning_model: An object of a contrastive model or a predictive model.
            encoder: Trainable pytorch model or list of models.
            pred_head: [Optional] Trainable pytoch model. If None, will use linear projection.
        '''
        
        pretrain_loader = DataLoader(self.dataset, batch_size, shuffle=True)
        p_optimizer = self.get_optim(self.p_optim)(encoder.parameters(), lr=self.p_lr, 
                                                   weight_decay=self.p_weight_decay)
        
        encoder = learning_model.train(encoder, pretrain_loader, self.p_optimizer, self.p_epochs)
        model = PredictionModel(encoder, pred_head, learning_model.dim, self.out_dim)
        
        for fold, (train_loader, test_loader, val_loader) in enumerate(zip(self.k_fold(fold_seed))):
            
            fold_model = copy.deepcopy(model)            
            f_optimizer = self.get_optim(f_optim)(model.parameters(), lr=p_lr, 
                                                  weight_decay=p_weight_decay)
            self.finetune(fold_model, f_optimizer, train_loader)
            
    
    def finetune(self, model, optimizer, loader):
        
        total_loss = 0
        correct = 0
        for data in loader:
            optimizer.zero_grad()
            data = data.to(torch.device('cuda:' + str(self.device)))
            out = model(data)
            loss = self.get_loss()(out, data.y.view(-1))
            pred = out.max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            loss.backward()
            optimizer.step()
            
        return correct / len(loader.dataset)
    
    
    def k_fold(self, seed):
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=seed)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(len(self.dataset)), self.dataset.data.y):
            test_indices.append(torch.from_numpy(idx))

        if epoch_select == 'test_max':
            val_indices = [test_indices[i] for i in range(self.n_folds)]
        else: # val_max
            val_indices = [test_indices[i - 1] for i in range(self.n_folds)]

        if self.label_rate < 1:

            label_skf = StratifiedKFold(int(1.0/self.label_rate), shuffle=True, random_state=seed)
            for i in range(folds):
                train_mask = torch.ones(len(self.dataset), dtype=torch.uint8)
                train_mask[test_indices[i].long()] = 0
                train_mask[val_indices[i].long()] = 0
                idx_train = train_mask.nonzero(as_tuple=False).view(-1)

                for _, idx in label_skf.split(torch.zeros(idx_train.size()[0]), 
                                              self.dataset.data.y[idx_train]):
                    idx_train = idx_train[idx]
                    break

                train_indices.append(idx_train)
        else:
            for i in range(folds):
                train_mask = torch.ones(len(self.dataset), dtype=torch.uint8)
                train_mask[test_indices[i].long()] = 0
                train_mask[val_indices[i].long()] = 0
                idx_train = train_mask.nonzero(as_tuple=False).view(-1)
                train_indices.append(idx_train)

        train_loader = DataLoader(self.dataset[train_indices], self.batch_size, shuffle=True)
        test_loader = DataLoader(self.dataset[test_indices], self.batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset[val_indices], self.batch_size, shuffle=True)
        return train_loader, test_loader, val_loader
        
    def get_optim(self, optim):
        
        optims = {'Adam': optim.Adam}
        
        return optims[optim]
    
    def get_loss(self):
        
        losses = {'clf': nn.functional.nll_loss}
        
        return losses[self.task]