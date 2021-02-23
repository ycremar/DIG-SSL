import torch
import torch.nn.functional as F


def JSE_loss(zs, pos=True, neg=False):
    '''
    Args:
        zs: List of two tensors of shape [z_dim]
    '''

    d_prime = torch.matmul(zs[0], zs[1])

    if pos:
        score = - F.softplus(-d_prime)

    elif neg:
        score = F.softplus(-d_prime) + d_prime
    else:
        raise Exception("Must define positive or negative pair!")

    # loss = mean(neg_scores) - mean(pos_scores)
    return score
