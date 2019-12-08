import torch
import torch.nn as nn
from model.utils import *

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
        
    def forward(self, xyz, features):
        xyz = xyz.permute(0, 2, 1)
        if features is not None:
            features = features.permute(0, 2, 1)

        if self.group_all:
            sampled_xyz, sampled_features = sample_and_group_all(xyz, features)
        else:
            sampled_xyz, sampled_features = sample_and_group(self.K, self.radius, self.num_sample, xyz, features)

        sampled_features = sampled_features.permute(0, 3, 2, 1)
        for blk in self.blks:
            sampled_features =  blk(sampled_features)

        sampled_features = torch.max(sampled_features, dim=2)[0]
        sampled_xyz = sampled_xyz.permute(0, 2, 1)
        return sampled_xyz, sampled_features
        

class PointNetMultiSetAbstraction(nn.Module):
    '''
    PointNet SA with multiple list of mlp \n
    Input: 
        K: numer of local regions
        radius: list of ball radius
        num_sample: list of number of samples
        in_channel: inout channel dimension
        mlp: list of layer width
        group_all: if True then produce a feature vector
    '''
    def __init__(self, K, radius_list, num_sample_list, in_channel, mlp_list):
        super(PointNetMultiSetAbstraction, self).__init__()
        self.K = K
        self.radius_list = radius_list
        self.num_sample_list = num_sample_list
        self.in_channel = in_channel
        self.mlp_list = mlp_list
        
        self.blks = nn.ModuleList()
        for i in range(len(self.mlp_list)):
            prev_channel = self.in_channel + 3
            blk = nn.ModuleList()
            for output_channel in self.mlp_list[i]:
                blk.append(
                    nn.Sequential(
                        nn.Conv2d(prev_channel, output_channel, 1),
                        nn.BatchNorm2d(output_channel),
                        nn.ReLU()
                    )
                )
                prev_channel = output_channel
            self.blks.append(blk)

    def forward(self, xyz, features):
        xyz = xyz.permute(0, 2, 1)
        if features is not None:
            features = features.permute(0, 2, 1)

        B, N, C = xyz.shape
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.K))
        new_features_list = []
        for i, radius in enumerate(self.radius_list):
            group_idx = query_ball_point(radius, self.num_sample_list[i], xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, self.K, 1, C)
            if features is not None:
                grouped_features = index_points(features, group_idx)
                grouped_features = torch.cat([grouped_features, grouped_xyz], dim=-1)
            else:
                grouped_features = grouped_xyz

            grouped_features = grouped_features.permute(0, 3, 2, 1)
            for blk in self.blks[i]:
                grouped_features = blk(grouped_features)
            
            new_features = torch.max(grouped_features, 2)[0]
            new_features_list.append(new_features)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_features_concat = torch.cat(new_features_list, dim=1)
        return new_xyz, new_features_concat


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

    def forward(self, xyz1, xyz2, feaures1, features2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        features2 = features2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interp_features = features2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists 
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1) 
            interp_features = torch.sum(index_features(features2, idx) * weight.view(B, N, 3, 1), dim=2)

        if feaures1 is not None:
            feaures1 = feaures1.permute(0, 2, 1)
            sampled_features = torch.cat([feaures1, interp_features], dim=-1)
        else:
            sampled_features = interp_features

        sampled_features = sampled_features.permute(0, 2, 1)

        for blk in self.blks:
            sampled_features =  blk(sampled_features)
        
        return sampled_features
