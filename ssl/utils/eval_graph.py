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
    
    def __init__(self, dataset, label_rate, out_dim, loss=nn.functional.nll_loss, 
                 epoch_select='test_max', metric='acc', n_folds=10, device=None):
        
        self.dataset = get_dataset(dataset)
        self.label_rate = label_rate
        self.out_dim = out_dim
        self.task = task
        self.metric = metric
        self.n_folds = n_folds
        self.device = device
        self.loss = loss
        self.epoch_select = epoch_select
        
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
        
        test_metrics = []
        val_losses = []
        val = (epoch_select == 'test_max' or epoch_select == 'test_min')
        for fold, (train_loader, test_loader, val_loader) in enumerate(zip(
            k_fold(self.n_folds, self.dataset, self.batch_size, self.label_rate, val, fold_seed))):
            
            fold_model = copy.deepcopy(model)            
            f_optimizer = self.get_optim(f_optim)(model.parameters(), lr=p_lr, 
                                                  weight_decay=p_weight_decay)
            self.finetune(fold_model, f_optimizer, train_loader)
            val_losses.append(self.eval_loss(fold_model, val_loader))
            test_metrics.append(self.eval_metric(fold_model, test_loader))
            
        val_loss, test_acc = torch.tensor(val_losses), torch.tensor(test_accs)
        val_loss = val_loss.view(self.n_folds, self.epochs)
        test_acc = test_acc.view(self.n_folds, self.epochs)

        if epoch_select == 'test_max':
            _, selection =  test_metrics.mean(dim=0).max(dim=0)
            selection = selection.repeat(self.n_folds)
        elif epoch_select == 'test_min':
            _, selection =  test_metrics.mean(dim=0).min(dim=0)
            selection = selection.repeat(self.n_folds)
        else:
            _, selection =  val_losses.min(dim=1)

        test_acc = test_acc[torch.arange(self.n_folds, dtype=torch.long), selection]
        test_acc_mean = test_acc.mean().item()
        test_acc_std = test_acc.std().item() 
        
        return test_acc_mean, test_acc_std
    
    
    def grid_search(self, learning_model, encoder, pred_head=None, fold_seed=12345,
                    p_lr_lst=[0.1,0.01,0.001,0.0001], p_epoch_lst=[20,40,60,80,100]):
        
        acc_m_lst = []
        acc_sd_lst = []
        paras = []
        for p_lr in p_lr_lst:
            for p_epoch in p_epoch_lst:
                self.setup_train_config(p_lr=p_lr, p_epoch=p_epoch)
                acc_m, acc_sd = self.evaluate(learning_model, encoder, pred_head, fold_seed)
                acc_m_lst.append(acc_m)
                acc_sd_lst.append(acc_sd)
                paras.append((p_lr, p_epoch))
        idx = np.argmax(acc_m)
        
        return acc_m[idx], acc_sd[idx], paras[idx]

    
    def finetune(self, model, optimizer, loader):
        
        total_loss = 0
        correct = 0
        for data in loader:
            optimizer.zero_grad()
            data = data.to(torch.device('cuda:' + str(self.device)))
            out = model(data)
            loss = self.loss(out, data.y.view(-1))
            pred = out.max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            loss.backward()
            optimizer.step()
            
        return correct / len(loader.dataset)
    
    
    def eval_loss(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()

        loss = 0
        for data in loader:
            data = data.to(torch.device('cuda:' + str(self.device)))
            with torch.no_grad():
                pred = model(data)
            loss += self.loss(pred, data.y.view(-1), reduction='sum').item()
            
        return loss / len(loader.dataset)
    
    
    def eval_acc(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()

        correct = 0
        for data in loader:
            data = data.to(torch.device('cuda:' + str(self.device)))
            with torch.no_grad():
                pred = model(data).max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            
        return correct / len(loader.dataset)
    
        
    def get_optim(self, optim):
        
        optims = {'Adam': optim.Adam}
        
        return optims[optim]
    
    def eval_metric(self, model, loader, eval_mode=True):
        
        if self.metric == 'acc':
            return self.eval_acc(model, loader, eval_mode)
        
    
    
def k_fold(n_folds, dataset, batch_size, label_rate=1, val=False, seed=12345):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if val:
        val_indices = [test_indices[i - 1] for i in range(n_folds)]
    else:
        val_indices = [test_indices[i] for i in range(n_folds)]

    if self.label_rate < 1:
        label_skf = StratifiedKFold(int(1.0/label_rate), shuffle=True, random_state=seed)
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
            train_mask = torch.ones(len(dataset), dtype=torch.uint8)
            train_mask[test_indices[i].long()] = 0
            train_mask[val_indices[i].long()] = 0
            idx_train = train_mask.nonzero(as_tuple=False).view(-1)
            train_indices.append(idx_train)

    train_loader = DataLoader(dataset[train_indices], batch_size, shuffle=True)
    test_loader = DataLoader(dataset[test_indices], batch_size, shuffle=True)
    val_loader = DataLoader(dataset[val_indices], batch_size, shuffle=True)

    return train_loader, test_loader, val_loader