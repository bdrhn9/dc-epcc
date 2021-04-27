import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['center_regressor','compute_center_loss','get_center_delta']
    
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


def compute_center_loss(features, centers, targets):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    # criterion = torch.nn.MSELoss()
    center_loss = F.mse_loss(features, target_centers)
    # center_loss = criterion(features, target_centers)
    return center_loss


def get_center_delta(features, centers, targets, alpha,device):
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.to(device)
    indices = indices.to(device)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).to(device).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result