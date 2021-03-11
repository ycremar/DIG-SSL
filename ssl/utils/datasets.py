import os.path as osp
import re

import torch
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from .feat_expansion import FeatureExpander


class TUDatasetExt(TUDataset):
    '''
    Used in GraphCL for feature expansion
    '''
    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'

    def __init__(self,
                root, 
                name,
                transform=None,
                pre_transform=None,
                pre_filter=None,
                use_node_attr=False,
                processed_filename='data.pt'
            ):
        self.processed_filename = processed_filename
        super(TUDatasetExt, self).__init__(root, name, transform, pre_transform, pre_filter, use_node_attr)

    @property
    def processed_file_names(self):
        return self.processed_filename


def get_dataset(name, sparse=True, feat_str="deg+ak3+reall", root=None):
    if name in ['DD']:
        feat_str = feat_str.replace('odeg100', 'odeg10')
        feat_str = feat_str.replace('ak3', 'ak1')


    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None

    pre_transform = FeatureExpander(degree=degree, onehot_maxdeg=onehot_maxdeg, AK=0).transform

    dataset = TUDatasetExp("./dataset/", name, pre_transform=pre_transform, use_node_attr=True, 
                        processed_filename="data_%s.pt" % feat_str)

    dataset.data.edge_attr = None

    return dataset
