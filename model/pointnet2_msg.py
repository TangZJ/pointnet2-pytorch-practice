import torch
import torch.nn as nn
from model.pointnet2_submodules import PointNetSetAbstraction as SA
from model.pointnet2_submodules import PointNetMultiSetAbstraction as MSA
from model.pointnet2_submodules import FullyConnectedLayer as FC
from model.pointnet2_submodules import PointNetFeaturePropagation as FP

class PointNet2ClsMSG(nn.Module):
    def __init__(self, num_class=40):
        super(PointNet2ClsMSG, self).__init__()
        self.sa1 = MSA(512, [0.1, 0.2, 0.4], [16, 32, 128], 0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = MSA(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = SA(in_channel=643, mlp=[256, 512, 1024], group_all=True)
        self.fc4 = FC(1024, 512, 0.4)
        self.fc5 = FC(512, 256, 0.4)
        self.fc6 = FC(256, num_class)

    def forward(self, xyz):
        xyz_, points_ = self.sa1(xyz, None)
        xyz_, points_ = self.sa2(xyz_, points_)
        xyz_, points_ = self.sa3(xyz_, points_)
        x = points_.view(xyz_.shape[0], 1024)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x