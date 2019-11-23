import torch
import torch.nn as nn
from model.utils import sample_and_group, sample_and_group_all, square_distance, index_points

class FullyConnectedLayer(nn.Module):
    '''
    PointNet FC \n
    '''
    def __init__(self, in_features, out_features, dropout_rate=None):
        super(FullyConnectedLayer, self).__init__()
        if dropout_rate is None:
            self.blks = nn.Linear(in_features, out_features)
        else:
            self.blks = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )

    def forward(self, x):
        return self.blks(x)


class PointNetSetAbstraction(nn.Module):
    '''
    PointNet SA \n
    Input: 
        K: numer of local regions
        radius: ball radius
        num_sample: number of samples
        in_channel: inout channel dimension
        mlp: list of layer width
        group_all: if True then produce a feature vector
    '''
    def __init__(self, K=1, radius=0.0, num_sample=0, in_channel=None, mlp=None, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.K = K
        self.radius = radius
        self.num_sample = num_sample
        self.in_channel = in_channel
        self.mlp = [in_channel] + mlp
        self.group_all = group_all
        
        self.blks = nn.ModuleList()
        for i in range(len(self.mlp)-1):
            self.blks.append(
                nn.Sequential(
                    nn.Conv2d(self.mlp[i], self.mlp[i+1], 1),
                    nn.BatchNorm2d(self.mlp[i+1]),
                    nn.ReLU()
                )
            )
        
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            sampled_xyz: sampled points position data, [B, C, S]
            sampled_points: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            sampled_xyz, sampled_points = sample_and_group_all(xyz, points)
        else:
            sampled_xyz, sampled_points = sample_and_group(self.K, self.radius, self.num_sample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        sampled_points = sampled_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        for blk in self.blks:
            sampled_points =  blk(sampled_points)

        sampled_points = torch.max(sampled_points, dim=2)[0]
        sampled_xyz = sampled_xyz.permute(0, 2, 1)
        return sampled_xyz, sampled_points
        

class PointNetFeaturePropagation(nn.Module):
    '''
    PointNet FP \n
    '''
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp = [in_channel] + mlp
        self.blks = nn.ModuleList()
        for i in range(len(self.mlp)-1):
            self.blks.append(
                nn.Sequential(
                    nn.Conv1d(self.mlp[i], self.mlp[i+1], 1),
                    nn.BatchNorm1d(self.mlp[i+1]),
                    nn.ReLU()
                )
            )

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            sampled_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            sampled_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            sampled_points = interpolated_points

        sampled_points = sampled_points.permute(0, 2, 1)

        for blk in self.blks:
            sampled_points =  blk(sampled_points)
        
        return sampled_points
