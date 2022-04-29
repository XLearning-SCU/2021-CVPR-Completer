import sys
import torch


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
    
#     Works with pytorch <= 1.2
#     p_i_j[(p_i_j < EPS).data] = EPS
#     p_j[(p_j < EPS).data] = EPS
#     p_i[(p_i < EPS).data] = EPS
    
    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device = p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device = p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device = p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss
