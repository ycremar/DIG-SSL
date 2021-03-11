import os
import os.path as osp
import shutil
import re

import torch
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from itertools import repeat, product
import numpy as np

from copy import deepcopy

from .feat_expansion import FeatureExpander


class TUDatasetExt(InMemoryDataset):
    '''
    Used in GraphCL for feature expansion
    '''
    # url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets')

    def __init__(self,
                root, 
                name,
                task,
                transform=None,
                pre_transform=None,
                pre_filter=None,
                use_node_attr=False,
                use_edge_attr=False,
                cleaned=False,
                processed_filename='data.pt'
            ):
        self.processed_filename = processed_filename
        super(TUDatasetExt, self).__init__(root, transform, pre_transform, pre_filter)

        self.task = task
        self.name = name
        self.cleaned = cleaned

        if self.task == "semisupervised":
            self.data, self.slices = torch.load(self.processed_paths[0])
            if self.data.x is not None and not use_node_attr:
                num_node_attributes = self.num_node_attributes
                self.data.x = self.data.x[:, num_node_attributes:]
            if self.data.edge_attr is not None and not use_edge_attr:
                num_edge_attributes = self.num_edge_attributes
                self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
    
        elif self.task == "unsupervised":
            self.data, self.slices = torch.load(self.processed_paths[0])
            if self.data.x is not None and not use_node_attr:
                num_node_attributes = self.num_node_attributes
                self.data.x = self.data.x[:, num_node_attributes:]
            if self.data.edge_attr is not None and not use_edge_attr:
                num_edge_attributes = self.num_edge_attributes
                self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
            if not (self.name == 'MUTAG' 
                or self.name == 'PTC_MR' 
                or self.name == 'DD' 
                or self.name == 'PROTEINS' 
                or self.name == 'NCI1' 
                or self.name == 'NCI109'
                ):
                edge_index = self.data.edge_index[0, :].numpy()
                _, num_edge = self.data.edge_index.size()
                nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
                nlist.append(edge_index[-1] + 1)

                num_node = np.array(nlist).sum()
                self.data.x = torch.ones((num_node, 1))

                edge_slice = [0]
                k = 0
                for n in nlist:
                    k = k + n
                    edge_slice.append(k)
                self.slices['x'] = torch.tensor(edge_slice)
        else:
            ValueError("Wrong task name")

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return self.processed_filename
        

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
    
    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[0], slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature


    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        
        if self.task == "unsupervised":
            node_num = data.edge_index.max()
            sl = torch.tensor([[n,n] for n in range(node_num)]).t()
            data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        return data



def get_dataset(name, task, sparse=True, feat_str="deg+ak3+reall", root=None):

    if task == "semisupervised":

        if name in ['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        if name in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')


        degree = feat_str.find("deg") >= 0
        onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
        onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None

        pre_transform = FeatureExpander(degree=degree, onehot_maxdeg=onehot_maxdeg, AK=0).transform

        dataset = TUDatasetExt("./dataset/", name, task, pre_transform=pre_transform, use_node_attr=True, 
                            processed_filename="data_%s.pt" % feat_str)
        
        dataset_pretrain = TUDatasetExt("./pretrain_dataset/", name, task, pre_transform=pre_transform, use_node_attr=True, 
                            processed_filename="data_%s.pt" % feat_str)

        dataset.data.edge_attr = None
        dataset_pretrain.data.edge_attr = None

        return dataset, dataset_pretrain

    elif task == "unsupervised":
        dataset = TUDatasetExt("./unsuper_dataset/", name=args.dataset, task=task)
        return dataset

    else:
        ValueError("Wrong task name")