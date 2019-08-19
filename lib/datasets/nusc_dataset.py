import os
import numpy as np
import copy
from torch.utils.data import Dataset

# nuScenes API
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as TRAIN_SCENE_NAMEs
from nuscenes.utils.splits import val as VAL_SCENE_NAMEs
from nuscenes.utils.data_classes import LidarPointCloud

class nuScenesDataset(Dataset):

    def __init__(self, dataroot, split, verbose=False):
        self.split = split
        if self.split in ['train', 'val']:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=verbose)
            scene_names = TRAIN_SCENE_NAMEs if self.split == 'train' else VAL_SCENE_NAMEs
        else:
            raise ValueError('Unsupported split type \'{}\''.format(split))
        
        self.sample_tokens = []
        for scene_info in self.nusc.scene:
            if scene_info['name'] in scene_names:
                sample_info = self.nusc.get('sample', scene_info['first_sample_token'])
                self.sample_tokens.append(sample_info['token'])
                while sample_info['next'] != '':
                    sample_info = self.nusc.get('sample', sample_info['next'])
                    self.sample_tokens.append(sample_info['token'])
    
    @property
    def dataroot(self):
        return self.nusc.dataroot
    
    def get_lidar(self, sample_info, nsweeps=1):
        lidar_info = self.nusc.get('sample_data', sample_info['data']['LIDAR_TOP'])
        if nsweeps == 1:
            lidar_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_info['filename']))
        else:
            lidar_pc, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample_info, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)
        lidar_ep_info = self.nusc.get('ego_pose', lidar_info['ego_pose_token'])
        lidar_cs_info = self.nusc.get('calibrated_sensor', lidar_info['calibrated_sensor_token'])
        return lidar_pc, lidar_ep_info, lidar_cs_info
    
    def get_lidar_triple(self, sample_info):
        curr_lidar_data = self.get_lidar(sample_info)

        prev_lidar_data = None
        if sample_info['prev'] != '':
            prev_sample_info = self.nusc.get('sample', sample_info['prev'])
            prev_lidar_data = self.get_lidar(prev_sample_info)
        
        next_lidar_data = None
        if sample_info['next'] != '':
            next_sample_info = self.nusc.get('sample', sample_info['next'])
            next_lidar_data = self.get_lidar(next_sample_info)
        
        return curr_lidar_data, prev_lidar_data, next_lidar_data
    
    def get_anns(self, sample_info):
        # have to be deepcopy for coordinate system transformation
        return [copy.deepcopy(self.nusc.get('sample_annotation', ann_token)) for ann_token in sample_info['anns']]
    
    def get_anns_triple(self, sample_info):
        curr_ann_infos = self.get_anns(sample_info)
        
        prev_ann_infos = None
        if sample_info['prev'] != '':
            prev_sample_info = self.nusc.get('sample', sample_info['prev'])
            prev_ann_infos = self.get_anns(prev_sample_info)
        
        next_ann_infos = None
        if sample_info['next'] != '':
            next_sample_info = self.nusc.get('sample', sample_info['next'])
            next_ann_infos = self.get_anns(next_sample_info)
        
        return curr_ann_infos, prev_ann_infos, next_ann_infos
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
