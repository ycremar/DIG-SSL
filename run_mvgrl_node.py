from sslgraph.utils import Encoder, get_node_dataset
from sslgraph.utils.eval_node import EvalUnsupevised
from sslgraph.contrastive.model import NodeMVGRL
import torch


if __name__ == '__main__':
    
    dataset, train_mask, val_mask, test_mask = get_node_dataset('cora')

    embed_dim = 512
    encoder_adj = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, 
                          n_layers=1, gnn='gcn', node_level=True, act='prelu')
    encoder_diff = Encoder(feat_dim=dataset[0].x.shape[1], hidden_dim=embed_dim, 
                           n_layers=1, gnn='gcn', node_level=True, act='prelu', edge_weight=True)
    mvgrl = NodeMVGRL(embed_dim, z_n_dim=embed_dim, diffusion_type='ppr', num_nodes=2000, device=1)

    evaluator = EvalUnsupevised(dataset, train_mask, val_mask, test_mask, device=1, log_interval=200)
    evaluator.setup_train_config(p_lr=0.001, p_epoch=2000)
    evaluator.evaluate(learning_model=mvgrl, encoder=[encoder_adj, encoder_diff])