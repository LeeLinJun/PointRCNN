# import sys
# sys.path.append('/home/srip19-pointcloud/linjun/pointcloud/work/multiframe/non-local/PointRCNN')

import torch
import torch.nn as nn
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from pointnet2_lib.pointnet2 import pytorch_utils as pt_utils

def get_model(input_channels=6, use_xyz=True):
    return Pointnet2MSG(input_channels=input_channels, use_xyz=use_xyz)


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=cfg.RPN.SA_CONFIG.NPOINTS[k],
                    radii=cfg.RPN.SA_CONFIG.RADIUS[k],
                    nsamples=cfg.RPN.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RPN.USE_BN
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()
        
        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )
        self.non_local_layer = NoneLocalLayer(input_channels=256,middle_channels=256//2,output_channels=256,bn=True)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        counter = 0
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
            counter += 1
            if counter ==3:
                self.non_local_layer()

        return l_xyz[0], l_features[0]


class NonLocalLayer(nn.Module):
    '''non-local network structure https://arxiv.org/pdf/1711.07971.pdf
    '''
    def __init__(self, input_channels, middle_channels, output_channels, bn):
        super().__init__()
        # input pointcloud should be in (B,C,T,N)
        # Conv1d need shape(B,C,N)
        self.theta = pt_utils.Conv2d(in_size=input_channels,
                                            out_size = middle_channels, bn=bn)  #(B,C,T,N) -> (B,C/2,T,N)
        self.phi = pt_utils.Conv2d(in_size=input_channels, out_size = middle_channels, bn=bn) 
        self.g = pt_utils.Conv2d(in_size=input_channels, out_size = middle_channels, bn=bn) 
        # recover the channel
        self.refine = pt_utils.Conv1d(in_size=middle_channels, out_size=output_channels, bn=bn) 
        # softmax
        self.softmax = torch.nn.Softmax(dim=1)     
    def forward(self,pc):
        # pc in shape (B,C,T,N)
        B,C,T,N = pc.shape
        # kernel
        theta = self.theta(pc).reshape([B,-1,T*N]).transpose(1,2) #(B,T*N,C/2)
        phi = self.phi(pc).reshape([B,-1,T*N]) #(B,C/2,T*N)
        g = self.g(pc).reshape([B,-1,T*N]).transpose(1,2) #(B,T*N,C/2)
        #(B,TN,C) x (B,C,TN) -> (B,TN,TN), 0~1 likelyhood along lines
        corr = self.softmax(torch.bmm(theta,phi))       
        # attention mul: (B,TN,TN) x (B,TN,C/2) -> (B,TN,C/2) ->(B,C/2,TN)
        y = torch.bmm(corr,g).transpose(1,2)        
        # refine and recover channel (B,TN,C/2) ->(B,TN,C) , res structure
        z= self.refine(y).reshape(B, T, N, -1).permute([0, 3, 1, 2]) + pc
        return z

def test_nl():
    pc = torch.ones([4,128,2,4096]).cuda()
    model = NonLocalLayer(input_channels=128,middle_channels =128//2,output_channels=128,bn=True).cuda()
    output = model(pc)
    return output
    