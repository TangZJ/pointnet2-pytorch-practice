import torch
import torch.nn as nn
from model.pointnet2_submodules import PointNetSetAbstraction as SA
from model.pointnet2_submodules import FullyConnectedLayer as FC
from model.pointnet2_submodules import PointNetFeaturePropagation as FP

class PointNet2ClsSSG(nn.Module):
    def __init__(self, num_class=40):
        super(PointNet2ClsSSG, self).__init__()
        self.sa1 = SA(K=512, radius=0.2, num_sample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = SA(K=128, radius=0.4, num_sample=64, in_channel=131, mlp=[128, 128, 256])
        self.sa3 = SA(in_channel=259, mlp=[256, 512, 1024], group_all=True)
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
    

class PointNet2SceneSegSSG(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2SceneSegSSG, self).__init__()
        self.sa1 = SA(1024, 0.1, 32, 6 + 3, [32, 32, 64])
        self.sa2 = SA(256, 0.2, 32, 64 + 3, [64, 64, 128])
        self.sa3 = SA(64, 0.4, 32, 128 + 3, [128, 128, 256])
        self.sa4 = SA(16, 0.8, 32, 256 + 3, [256, 256, 512])
        self.fp5 = FP(768, [256, 256])
        self.fp6 = FP(384, [256, 256])
        self.fp7 = FP(320, [256, 128])
        self.fp8 = FP(128, [128, 128, 128])
        self.conv9 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )
        
    def forward(self, xyz, points):
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp5(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp6(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp7(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp8(xyz, l1_xyz, None, l1_points)

        x = self.conv9(l0_points)

        return x