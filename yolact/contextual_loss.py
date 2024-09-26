import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualLoss(nn.Module):
    def __init__(self, band_width=0.5):
        super(ContextualLoss, self).__init__()
        self.band_width = band_width

    def forward(self, feature1, feature2):
        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)

        dist_matrix = torch.matmul(feature1, feature2.t())
        dist_matrix = torch.clamp(dist_matrix, min=-1, max=1)

        dist_matrix = (1 - dist_matrix) / 2
        kernel_matrix = torch.exp(-dist_matrix / self.band_width)
        
        sum_kernel = torch.sum(kernel_matrix, dim=1)
        contextual_loss = -torch.log(torch.diag(kernel_matrix) / (sum_kernel - torch.diag(kernel_matrix) + 1e-5))

        return contextual_loss.mean()
