import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
from scipy.stats import wasserstein_distance
from geomloss import SamplesLoss
from layers import SinkhornDistance
"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.pos_margin = opt.loss_contrastive_pos_margin
        self.neg_margin = opt.loss_contrastive_neg_margin
        self.batchminer = batchminer

        self.name           = 'wcontrastive'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
    def forward(self, batch, labels, **kwargs):
        sampled_triplets = self.batchminer(batch, labels)
        anchors   = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]
        wloss = SamplesLoss("sinkhorn", p = 2, blur=0.05, scaling = .99, backend = "online")
#       pos_dists = torch.mean(F.relu(wasserstein_distance(batch[anchors,:].cpu().detach().numpy(), batch[positives,:].cpu().detach().numpy()) -  self.pos_margin))
#       neg_dists = torch.mean(F.relu(self.neg_margin - wasserstein_distance(batch[anchors,:].cpu().detach().numpy(), batch[negatives,:].cpu().detach().numpy())))
#        print("types")
#        print(type(batch[anchors,:])
#       print(type(wloss(batch[anchors,:], batch[positives,:])))
#        pos_dists = torch.mean(F.relu(torch.from_numpy(np.array(wloss(batch[anchors,:], batch[positives,:]).item() -  self.pos_margin)))))
#        neg_dists = torch.mean(F.relu(torch.from_numpy(np.array(self.neg_margin - wloss(batch[anchors,:], batch[negatives,:]).item()))))
#        sinkhorn = SinkhornDistance(eps = 0.1, max_iter = 100, reduction=None)
        dist1 = wloss(batch[anchors,:], batch[positives,:])
        dist2 = wloss(batch[anchors,:], batch[negatives,:])
        pos_dists = torch.mean(F.relu(dist1-self.pos_margin))
        neg_dists = torch.mean(F.relu(self.neg_margin - dist2))
        loss      = pos_dists + neg_dists
        return loss
