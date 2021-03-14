import sys, copy, torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    

class EvalUnsupevised(object):
    
    def __init__(self, dataset, out_dim, 
                 epoch_select='test_max', metric='acc', n_folds=10, device=None):
        
        self.dataset = dataset
        self.epoch_select = epoch_select
        self.metric = metric
        self.n_folds = n_folds
        self.out_dim = out_dim
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Use default config if not further specified
        self.setup_train_config()

    def setup_train_config(self, batch_size = 256,
                           p_optim = 'Adam', p_lr = 0.0001, p_weight_decay = 0, p_epoch = 100,
                           f_optim = 'SVC', f_search=True):
        
        self.batch_size = batch_size

        self.p_optim = p_optim
        self.p_lr = p_lr
        self.p_weight_decay = p_weight_decay
        self.p_epoch = p_epoch

        self.f_optim = f_optim
        self.f_search = f_search
    
    def evaluate(self, learning_model, encoder, pred_head=None, fold_seed=12345):
        '''
        Args:
            learning_model: An object of a contrastive model or a predictive model.
            encoder: List or trainable pytorch model.
            pred_head: [Optional] Trainable pytoch model. If None, will use linear projection.
        '''
        
        pretrain_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        p_optimizer = self.get_optim(self.p_optim)(encoder.parameters(), lr=self.p_lr, 
                                                   weight_decay=self.p_weight_decay)
        
        encoder = learning_model.train(encoder, pretrain_loader, p_optimizer, self.p_epoch)
        model = PredictionModel(encoder, pred_head, learning_model.z_dim, self.out_dim)
        
        test_scores = []
        val = not (self.epoch_select == 'test_max' or self.epoch_select == 'test_min')
        for fold, train_loader, test_loader, val_loader in \
            k_fold(self.n_folds, self.dataset, self.batch_size, 1, val, fold_seed):
            fold_model = copy.deepcopy(model)
            test_score = self.get_optim(self.f_optim)(fold_model, f_optimizer, (train_loader, test_loader))
            test_scores.append(test_score)

        test_scores = torch.tensor(test_scores)
        test_score_mean = test_scores.mean().item()
        test_score_std = test_scores.std().item() 
        
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

    
    def svc_clf(self, model, loader):

        model.eval()
        
        total_acc = 0
        total_num = 0

        for train_data, test_data in zip(loader):

            train_data = train_data.to(torch.device('cuda:' + str(self.device)))
            test_data = test_data.to(torch.device('cuda:' + str(self.device)))

            x_train = model(train_data)
            y_train = preprocessing.LabelEncoder().fit_transform(train_data.y.cpu())
            x_test = model(test_data)
            y_test = preprocessing.LabelEncoder().fit_transform(test_data.y.cpu())
            
            x_train, y_train = x_train.cpu().detach().numpy(), np.array(y_train)
            x_test, y_test = x_test.cpu().detach().numpy(), np.array(y_test)

            if self.f_search:
                params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
                classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
            else:
                classifier = SVC(C=10)

            classifier.fit(x_train, y_train)
            acc = accuracy_score(y_test, classifier.predict(x_test))

            total_acc += acc * x_train.size(0)
            total_num += x_train.size(0)
            
        return total_acc / total_num
    
    
    def log_clf(self, model, loader):
        
        model.eval()
        
        total_acc = 0
        total_num = 0

        for train_data, test_data in zip(loader):

            train_data = train_data.to(torch.device('cuda:' + str(self.device)))
            test_data = test_data.to(torch.device('cuda:' + str(self.device)))

            with torch.no_grad():
                train_embs, test_embs = model(train_data).detach(), model(test_data).detach()
                
            train_lbls = np.array(preprocessing.LabelEncoder().fit_transform(train_data.y.cpu()))
            test_lbls = np.array(preprocessing.LabelEncoder().fit_transform(test_data.y.cpu()))

            train_lbls, test_lbls= torch.from_numpy(train_lbls).cuda(), torch.from_numpy(test_lbls).cuda()

            log = LogReg(hid_units, nb_classes)
            log.cuda()
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

            best_val = 0
            test_acc = None
            for it in range(100):
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward()
                opt.step()

            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs_val.append(acc.item())
            
        return total_acc / total_num
    
        
    def get_optim(self, optim):
        
        if optim == 'SVC':
            return self.svc_clf
        elif optim == 'LogReg':
            return self.log_clf
        else:
            return None
    

    
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
        
        return nn.functional.log_softmax(out, dim=-1)

    

class EvalSemisupevised(object):
    
    def __init__(self, dataset, dataset_pretrain, label_rate, out_dim, loss=nn.functional.nll_loss, 
                 epoch_select='test_max', metric='acc', n_folds=10, device=None):
        
        self.dataset, self.dataset_pretrain = dataset, dataset_pretrain
        self.label_rate = label_rate
        self.out_dim = out_dim
        self.metric = metric
        self.n_folds = n_folds
        self.loss = loss
        self.epoch_select = epoch_select
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
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
        
        self.f_optim = f_optim
        self.f_lr = f_lr
        self.f_weight_decay = f_weight_decay
        self.f_epoch = f_epoch
        
    
    def evaluate(self, learning_model, encoder, pred_head=None, fold_seed=12345):
        '''
        Args:
            learning_model: An instance of a contrastive model or a predictive model.
            encoder: Trainable pytorch model or list of models.
            pred_head: [Optional] Trainable pytoch model. If None, will use linear projection.
        '''
        pretrain_loader = DataLoader(self.dataset_pretrain, self.batch_size, shuffle=True)
        p_optimizer = self.get_optim(self.p_optim)(encoder.parameters(), lr=self.p_lr,
                                                   weight_decay=self.p_weight_decay)
        if self.p_epoch > 0:
            encoder = learning_model.train(encoder, pretrain_loader, p_optimizer, self.p_epoch)
        model = PredictionModel(encoder, pred_head, learning_model.z_dim, self.out_dim).to(self.device)
        
        test_scores = []
        val_losses = []
        val = not (self.epoch_select == 'test_max' or self.epoch_select == 'test_min')
        
        for fold, train_loader, test_loader, val_loader in k_fold(
            self.n_folds, self.dataset, self.batch_size, self.label_rate, val, fold_seed):
            
            fold_model = copy.deepcopy(model)
            f_optimizer = self.get_optim(self.f_optim)(fold_model.parameters(), lr=self.f_lr,
                                                       weight_decay=self.f_weight_decay)
            with trange(self.f_epoch) as t:
                for epoch in t:
                    t.set_description('Fold %d, finetuning' % (fold+1))
                    self.finetune(fold_model, f_optimizer, train_loader)
                    val_loss = self.eval_loss(fold_model, val_loader)
                    test_score = self.eval_metric(fold_model, test_loader)
                    val_losses.append(val_loss)
                    test_scores.append(test_score)

                    t.set_postfix(val_loss='{:.4f}'.format(val_loss), 
                                  acc='{:.4f}'.format(test_score))
        

        val_losses, test_scores = torch.tensor(val_losses), torch.tensor(test_scores)
        val_losses = val_losses.view(self.n_folds, self.f_epoch)
        test_scores = test_scores.view(self.n_folds, self.f_epoch)

        if self.epoch_select == 'test_max':
            _, selection =  test_scores.mean(dim=0).max(dim=0)
            selection = selection.repeat(self.n_folds)
        elif self.epoch_select == 'test_min':
            _, selection =  test_scores.mean(dim=0).min(dim=0)
            selection = selection.repeat(self.n_folds)
        else:
            _, selection =  val_losses.min(dim=1)

        test_scores = test_scores[torch.arange(self.n_folds, dtype=torch.long), selection]
        test_acc_mean = test_scores.mean().item()
        test_acc_std = test_scores.std().item() 
        
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
        
        model.train()
        for data in loader:
            optimizer.zero_grad()
            data = data.to(self.device)
            out = model(data)
            loss = self.loss(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
                
    
    def eval_loss(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()

        loss = 0
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = model(data)
            loss += self.loss(pred, data.y.view(-1), reduction='sum').item()
            
        return loss / len(loader.dataset)
    
    
    def eval_acc(self, model, loader, eval_mode=True):
        
        if eval_mode:
            model.eval()

        correct = 0
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = model(data).max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            
        return correct / len(loader.dataset)
    
        
    def get_optim(self, optim):
        
        optims = {'Adam': torch.optim.Adam}
        
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

    if label_rate < 1:
        label_skf = StratifiedKFold(int(1.0/label_rate), shuffle=True, random_state=seed)
        for i in range(n_folds):
            train_mask = torch.ones(len(dataset), dtype=torch.uint8)
            train_mask[test_indices[i].long()] = 0
            train_mask[val_indices[i].long()] = 0
            idx_train = train_mask.nonzero(as_tuple=False).view(-1)

            for _, idx in label_skf.split(torch.zeros(idx_train.size()[0]), 
                                          dataset.data.y[idx_train]):
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
            
    for i in range(n_folds):
        train_loader = DataLoader(dataset[train_indices[i]], batch_size, shuffle=True)
        test_loader = DataLoader(dataset[test_indices[i]], batch_size, shuffle=False)
        val_loader = DataLoader(dataset[val_indices[i]], batch_size, shuffle=False)

        yield i, train_loader, test_loader, val_loader