import torch
import torch.nn as nn

__all__ = ['center_regressor']
    
class center_regressor(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        
        for binary and one-vs-rest situation set mutex_label = True
        mutex_label (bool): indicates given labels is -1 or 1
    """
    def __init__(self, feat_dim=2,num_classes=10, mutex_label = True):
        super(center_regressor, self).__init__()
        self.num_classes = num_classes
        self.mutex_label = mutex_label
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().cuda()
        if(self.mutex_label):
            mask = labels>=0
        else:
            labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).mean() / batch_size

        return loss

