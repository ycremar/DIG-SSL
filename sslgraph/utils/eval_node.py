import sys, copy, torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
from torch import optim
from sslgraph.utils import get_node_dataset


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


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


class EvalUnsupevised(object):
    def __init__(self, dataset, out_dim, metric='acc', device=None):

        self.dataset_pretrain = get_node_dataset(dataset, task="unsupervised", mode='pretrain')
        self.dataset_train = get_node_dataset(dataset, task="unsupervised", mode='train')
        self.dataset_val = get_node_dataset(dataset, task="unsupervised", mode='val')
        self.dataset_test = get_node_dataset(dataset, task="unsupervised", mode='test')
        self.metric = metric
        self.device = device
        self.out_dim = out_dim

        # Use default config if not further specified
        self.setup_train_config()

    def setup_train_config(self, p_optim='Adam', p_lr=0.001, p_weight_decay=0.0, p_epoch=3000, p_loss=nn.BCEWithLogitsLoss(),
                           f_optim='Adam', f_lr=0.01, f_weight_decay=0.0, f_epoch=100, f_loss = nn.CrossEntropyLoss()):

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch
        self.p_loss = p_loss

        self.f_optim = f_optim
        self.f_lr = f_lr
        self.f_weight_decay = f_weight_decay
        self.f_epoch = f_epoch
        self.f_loss = f_loss

    def grid_search(self, learning_model, encoder, seed=39,
                    p_lr_lst=None, p_epoch_lst=None):
        if p_lr_lst is None:
            p_lr_lst = [0.001]
        if p_epoch_lst is None:
            p_epoch_lst = [20, 40, 60, 80, 100]

        acc_m_lst = []
        acc_sd_lst = []
        paras = []
        for p_lr in p_lr_lst:
            for p_epoch in p_epoch_lst:
                self.setup_train_config(p_lr=p_lr, p_epoch=p_epoch)
                acc_m, acc_sd = self.evaluate(learning_model, encoder, seed)
                acc_m_lst.append(acc_m)
                acc_sd_lst.append(acc_sd)
                paras.append((p_lr, p_epoch))
        idx = np.argmax(acc_m_lst)[0]
        return acc_m_lst[idx], acc_sd_lst[idx], paras[idx]

    def evaluate(self, learning_model, encoder, seed=39):
        '''
        Args:
            learning_model: An object of a contrastive model or a predictive model.
            encoder: Trainable pytorch model or list of models.
            pred_head: [Optional] Trainable pytoch model. If None, will use linear projection.
        '''

        pretrain_loader = DataLoader(self.dataset_pretrain, batch_size=1)
        p_optimizer = self.get_optim(self.p_optim)(encoder.parameters(), lr=self.p_lr,
                                                   weight_decay=self.p_weight_decay)

        encoder = learning_model.train(encoder, pretrain_loader, p_optimizer, self.p_epochs)
        self.embeds, nb_classes = self.embedding(encoder, pretrain_loader)

        train_loader = DataLoader(self.dataset_train, batch_size=1)
        val_loader = DataLoader(self.dataset_val, batch_size=1)
        test_loader = DataLoader(self.dataset_test, batch_size=1)

        val_metrics = []
        test_metrics = []
        for _ in range(50):
            log = LogReg(self.embeds.shape[1], nb_classes)
            f_optimizer = self.get_optim(self.f_optim)(log.parameters(), lr=self.f_lr, weight_decay=self.f_weight_decay)
            log.to(torch.device(self.device))

            for _ in range(self.f_epoch):
                self.finetune(log, f_optimizer, train_loader)
            val_metrics.append(self.eval_metric(log, val_loader))
            test_metrics.append(self.eval_metric(log, test_loader))

        test_acc_mean = test_metrics.mean().item()
        test_acc_std = test_metrics.std().item()
        return test_acc_mean, test_acc_std

    def finetune(self, model, optimizer, loader):
        for idx_train, train_lbls in loader:
            train_embs = self.embeds[idx_train]
            model.train()
            optimizer.zero_grad()
            logits = model(train_embs)
            loss = self.f_loss(logits, train_lbls)
            loss.backward()
            optimizer.step()

    def embedding(self, encoder, loader):
        encoder.eval()
        for data, nb_classes in loader:
            embeds = encoder(data)
            return embeds, nb_classes

    def eval_metric(self, model, loader, eval_mode=True):
        if self.metric == 'acc':
            return self.eval_acc(model, loader, eval_mode)

    def eval_acc(self, model, loader, eval_mode=True):
        if eval_mode:
            model.eval()
        for idx, lbls in loader:
            embs = self.embeds[idx]
            with torch.no_grad():
                logits = model(embs)
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == lbls).float() / lbls.shape[0]
        return acc

    def get_optim(self, optim):

        optims = {'Adam': optim.Adam}

        return optims[optim]


class EvalSemisupevised(object):
    
    def __init__(self, dataset, label_rate, out_dim, 
                 task='clf', metric='acc', n_folds=10, device=None):
        
        self.dataset = dataset
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