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
        self.non_local_layer = NonLocalLayer(input_channels=512,middle_channels=512//2,output_channels=512,bn=True)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        f1 = pointcloud[:,:cfg.RPN.NUM_POINTS,:3]
        f0 = pointcloud[:,cfg.RPN.NUM_POINTS:,:3]
        # frame 0
        xyz_f0, features_f0 = self._break_up_pc(f0)
        l_xyz_f0, l_features_f0 = [xyz_f0], [features_f0]
        # frame 1
        xyz_f1, features_f1 = self._break_up_pc(f1)
        l_xyz_f1, l_features_f1 = [xyz_f1], [features_f1]

        for i in range(len(self.SA_modules)):
            #f0
            li_xyz_f0, li_features_f0 = self.SA_modules[i](l_xyz_f0[i], l_features_f0[i])
            l_xyz_f0.append(li_xyz_f0)
            l_features_f0.append(li_features_f0)
            #f1
            li_xyz_f1, li_features_f1 = self.SA_modules[i](l_xyz_f1[i], l_features_f1[i])
            l_xyz_f1.append(li_xyz_f1)
            l_features_f1.append(li_features_f1)

        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        for i in range(-1, -(len(self.FP_modules)- 1), -1): #end with i=-2
            l_features_f0[i - 1] = self.FP_modules[i](
                l_xyz_f0[i - 1], l_xyz_f0[i], l_features_f0[i - 1], l_features_f0[i]
            )
            l_features_f1[i - 1] = self.FP_modules[i](
                l_xyz_f1[i - 1], l_xyz_f1[i], l_features_f1[i - 1], l_features_f1[i]
            )   
        fusion_features = self.non_local_layer(
            torch.cat([l_features_f0[i-1].unsqueeze(2),
            l_features_f1[i-1].unsqueeze(2)], dim=2)
            )
        l_features_f1[i - 1], l_features_f0[i - 1] = fusion_features[:, :, 1, :].contiguous(), fusion_features[:, :, 0, :].contiguous()
        
        i -= 1 # i=-3 now
        l_features_f0[i - 1] = self.FP_modules[i](
                l_xyz_f0[i - 1], l_xyz_f0[i], l_features_f0[i - 1], l_features_f0[i]
        )
        l_features_f1[i - 1] = self.FP_modules[i](
                l_xyz_f1[i - 1], l_xyz_f1[i], l_features_f1[i - 1], l_features_f1[i]
        )

        i -= 1 # i=-4 now
        l_features_f0[i - 1] = self.FP_modules[i](
                l_xyz_f0[i - 1], l_xyz_f0[i], l_features_f0[i - 1], l_features_f0[i]
        )
        l_features_f1[i - 1] = self.FP_modules[i](
                l_xyz_f1[i - 1], l_xyz_f1[i], l_features_f1[i - 1], l_features_f1[i]
        )
        return torch.cat([l_xyz_f1[0],l_xyz_f0[0]],dim=1), torch.cat([l_features_f1[0],l_features_f0[0]],dim=2)


class NonLocalLayer(nn.Module):
    '''non-local network structure https://arxiv.org/pdf/1711.07971.pdf
    '''
    def __init__(self, input_channels, middle_channels, output_channels, bn):
        super().__init__()
        # input pointcloud should be in (B,C,T,N)
        # Conv1d need shape(B,C,N),Conv2d need shape(B,C,H,W)
        self.theta = pt_utils.Conv2d(in_size=input_channels, out_size=middle_channels, activation=None, bn=bn)  #(B,C,T,N) -> (B,C/2,T,N)
        self.phi = pt_utils.Conv2d(in_size=input_channels, out_size=middle_channels, activation=None, bn=bn) #(B,C,T,N) -> (B,C/2,T,N)
        self.g = pt_utils.Conv2d(in_size=input_channels, out_size=middle_channels, bn=bn) #(B,C,T,N) -> (B,C/2,T,N)
        # recover the channel
        self.refine = pt_utils.Conv2d(in_size=middle_channels, out_size=output_channels, bn=bn) #(B,C/2,T,N,) -> (B,C,T,N)
        # softmax
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self,pc):
        # pc in shape (B,C,T,N)
        B,C,T,N = pc.shape
        # kernel
        theta = self.theta(pc).reshape([B,-1,T*N]).transpose(1,2) #(B,T*N,C/2)
        phi = self.phi(pc).reshape([B,-1,T*N]) #(B,C/2,T*N)
        g = self.g(pc).reshape([B,-1,T*N]).transpose(1,2) #(B,T*N,C/2)
        #(B,TN,C/2) x (B,C/2,TN) -> (B,TN,TN), 0~1 likelyhood along lines
        corr = self.softmax(torch.bmm(theta,phi))       
        # attention mul: (B,TN,TN) x (B,TN,C/2) -> (B,TN,C/2) ->(B,C/2,TN) ->(B,C/2,T,N)
        y = torch.bmm(corr,g).transpose(1,2).reshape(B,-1, T, N)   
        # refine and recover channel (B,C/2,T,N) ->(B,C,T,N)  ---+pc---(res structure) -> (B,C,T,N)
        z= self.refine(y) + pc
        return z

# def test_nl():
#     pc = torch.ones([4,128,2,4096]).cuda()
#     model = NonLocalLayer(input_channels=128,middle_channels =128//2,output_channels=128,bn=True).cuda()
#     output = model(pc)
#     return output
    