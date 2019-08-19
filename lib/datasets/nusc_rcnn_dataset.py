import os
import numpy as np
import torch
from pyquaternion import Quaternion
from lib.datasets.nusc_dataset import nuScenesDataset
import lib.utils.kitti_utils as kitti_utils
from lib.config import cfg

class nuScenesRCNNDataset(nuScenesDataset):

    def __init__(self, dataroot, split, mode, classes='all', verbose=False, npoints=16384, random_select=True,
                 logger=None):
        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        super(nuScenesRCNNDataset, self).__init__(dataroot=dataroot, split=split, verbose=verbose)
        
        self.mode = mode
        if classes == 'all':
            self.classes = ('background', 'barrier', 'trafficcone', 'bicycle', 'motorcycle', 
                            'pedestrian', 'car', 'bus', 'construction', 'trailer', 'truck')
        elif classes == 'car':
            self.classes = ('background', 'car')
        else:
            ValueError("Invalid classes: %s" % classes)
        self.num_class = self.classes.__len__()
        self.random_select = random_select
        self.npoints = npoints
        self.logger = logger

        if not self.random_select:
            self.logger.warning('random select is False')
        
        if cfg.RPN.ENABLED:
            self.logger.info('Loading %s samples ... ' % self.mode)
            if self.mode == 'TRAIN':
                self.preprocess_rpn_training_data()
            else:
                self.logger.info('Done: total %s samples %d' % (self.mode, len(self.sample_tokens)))
        elif cfg.RCNN.ENABLED:
            self.preprocess_rpn_training_data()
    
    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_token is stored in self.sample_tokens
        """
        valid_tokens = []
        for sample_token in self.sample_tokens:
            sample_ann_infos = self.filterate_anns(self.get_anns(self.nusc.get('sample', sample_token)), add_label=False)
            if sample_ann_infos.__len__() > 0:
                valid_tokens.append(sample_token)

        self.logger.info('Done: filter valid %s samples: %d / %d\n' % (self.mode, len(valid_tokens), len(self.sample_tokens)))
        self.sample_tokens = valid_tokens
        return
    
    def __len__(self):
        return self.sample_tokens.__len__()
    
    def __getitem__(self, index):
        sample_info = self.nusc.get('sample', self.sample_tokens[index])
        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(sample_info)
        else:
            raise NotImplementedError('Offline rcnn training not supported in nuScenes.')
    
    @staticmethod
    def remove_useless_points(pc):
        # remove close points
        x_range = np.abs(pc.points[0]) < 1.0
        y_range = np.abs(pc.points[1]) < 2.5
        pc.points = pc.points[:, np.logical_not(np.logical_and(x_range, y_range))]
        # remove far away points
        dist = np.linalg.norm(pc.points[:2], axis=0)
        pc.points = pc.points[:, dist < 55.0]
        return pc

    @staticmethod
    def anns_to_bboxes(ann_infos):
        if ann_infos.__len__() > 0:
            ann_bboxes = []
            ann_labels = []
            for ann_info in ann_infos:
                yaw = np.arccos(ann_info['rotation'][0]) * 2.0
                yaw = yaw if ann_info['rotation'][3] >= 0.0 else -yaw
                if yaw > np.pi:
                    yaw -= 2 * np.pi
                elif yaw < -np.pi:
                    yaw += 2 * np.pi
                # change to KITTI format
                kitti_trans = [ann_info['translation'][0], -ann_info['translation'][2], ann_info['translation'][1]]
                kitti_size = [ann_info['size'][2], ann_info['size'][0], ann_info['size'][1]] # (h, w, l)
                kitti_trans[1] += kitti_size[0] / 2.0
                kitti_ry = -yaw
                ann_bboxes.append(kitti_trans + kitti_size + [kitti_ry])
                ann_labels.append(ann_info['category_label'])
            ann_bboxes = np.array(ann_bboxes) # (N, 7)
            ann_labels = np.array(ann_labels) # (N)
        else:
            ann_bboxes = np.zeros((0, 7))
            ann_labels = np.zeros((0))

        return ann_bboxes, ann_labels
    
    @staticmethod
    def pc_trans(lidar_pc, lidar_ep_info, lidar_cs_info):
        rot1 = Quaternion(lidar_cs_info['rotation'])
        rot2 = Quaternion(lidar_ep_info['rotation'])
        lidar_pc.rotate(rot1.rotation_matrix)
        lidar_pc.translate(lidar_cs_info['translation'])
        lidar_pc.rotate(rot2.rotation_matrix)
        lidar_points = lidar_pc.points[:3].T # (N, 3)
        lidar_intensity = lidar_pc.points[3].reshape(-1, 1) # (N, 1) # currently don't know how to use it
        return lidar_points, lidar_intensity
    
    @staticmethod
    def ann_trans(ann_infos, ref_ep_info):
        for ann_info in ann_infos:
            for dim in range(ann_info['translation'].__len__()):
                ann_info['translation'][dim] -= ref_ep_info['translation'][dim]
        return ann_infos

    def filterate_anns(self, ann_infos, min_pts=10, add_label=True):
        valid_ann_infos = []
        for ann_info in ann_infos:
            # remove super sparse anns
            if ann_info['num_lidar_pts'] < min_pts:
                continue
            # filterate and change category names
            for label in range(1, self.num_class):
                if self.classes[label] in ann_info['category_name']:
                    if add_label:
                        ann_info['category_label'] = label
                    valid_ann_infos.append(ann_info)
                    break
        return valid_ann_infos
    
    def get_rpn_sample(self, sample_info):
        
        # get original data and preprocess point cloud and labels
        sample_outputs = {'token': sample_info['token'], 'random_select': self.random_select}
        sample_lidar_pc, sample_lidar_ep_info, sample_lidar_cs_info = self.get_lidar(sample_info)
        sample_lidar_pc = self.remove_useless_points(sample_lidar_pc)
        sample_ann_infos = self.filterate_anns(self.get_anns(sample_info))

        # point cloud coordinate transform for better usage of labels
        sample_lidar_points, sample_lidar_intensity = self.pc_trans(sample_lidar_pc, sample_lidar_ep_info, sample_lidar_cs_info)

        # annotation translate
        sample_ann_infos = self.ann_trans(sample_ann_infos, sample_lidar_ep_info)

        # points sampling
        if self.mode == 'TRAIN' or self.random_select:
            if self.npoints < sample_lidar_points.__len__():
                sample_lidar_points_dist = np.linalg.norm(sample_lidar_points[:, :2], axis=1)
                sample_lidar_points_isfar = sample_lidar_points_dist > 15.0
                far_idxs = np.where(sample_lidar_points_isfar)[0]
                near_idxs = np.where(sample_lidar_points_isfar == False)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - far_idxs.__len__(), replace=False)
                idxs_choice = np.concatenate((near_idxs_choice, far_idxs))
            else:
                idxs_choice = np.arange(sample_lidar_points.__len__())
                if self.npoints > sample_lidar_points.__len__():
                    extra_choice = np.random.choice(idxs_choice, self.npoints - sample_lidar_points.__len__(), replace=False)
                    idxs_choice = np.concatenate((idxs_choice, extra_choice))
            np.random.shuffle(idxs_choice)
            sample_lidar_points = sample_lidar_points[idxs_choice] # (npoints, 3)
            sample_lidar_intensity = sample_lidar_intensity[idxs_choice] # (npoints, 1)
        
        # change to KITTI axis
        sample_lidar_points = sample_lidar_points[:, [0, 2, 1]]
        sample_lidar_points[:, 1] *= -1.0
        
        if self.mode == 'TEST':
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((sample_lidar_points, sample_lidar_intensity), axis=1)  # (N, C)
            else:
                pts_input = sample_lidar_points
            sample_outputs['pts_input'] = pts_input.astype(np.float32)
            sample_outputs['pts_rect'] = sample_lidar_points.astype(np.float32)
            sample_outputs['pts_features'] = sample_lidar_intensity.astype(np.float32)
            return sample_outputs

        # gether annotations
        sample_ann_bboxes, sample_ann_labels = self.anns_to_bboxes(sample_ann_infos)

        # data augmentation, points and bboxes changed
        if cfg.AUG_DATA:
            sample_lidar_points, sample_ann_bboxes, aug_method = self.data_augmentation(sample_lidar_points, sample_ann_bboxes)
            sample_outputs['aug_method'] = aug_method
        
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((sample_lidar_points, sample_lidar_intensity), axis=1)  # (N, C)
        else:
            pts_input = sample_lidar_points

        sample_outputs['pts_input'] = pts_input.astype(np.float32)
        sample_outputs['pts_rect'] = sample_lidar_points.astype(np.float32)
        sample_outputs['pts_features'] = sample_lidar_intensity.astype(np.float32)
        sample_outputs['gt_boxes3d'] = sample_ann_bboxes.astype(np.float32)
        sample_outputs['gt_labels'] = sample_ann_labels.astype(np.int32)

        if not cfg.RPN.FIXED:
            rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(sample_lidar_points, sample_ann_bboxes, sample_ann_labels)
            sample_outputs['rpn_cls_label'] = rpn_cls_label.astype(np.int32)
            sample_outputs['rpn_reg_label'] = rpn_reg_label.astype(np.float32)

        return sample_outputs
    
    @staticmethod
    def generate_rpn_training_labels(points, bboxes, labels):
        cls_label = np.zeros((points.__len__()))
        reg_label = np.zeros((points.__len__(), 7))
        corners = kitti_utils.boxes3d_to_corners3d(bboxes)
        extend_bboxes = kitti_utils.enlarge_box3d(bboxes, extra_width=0.1)
        extend_corners = kitti_utils.boxes3d_to_corners3d(extend_bboxes)
        for k in range(bboxes.__len__()):
            box_corners = corners[k]
            fg_pt_flag = kitti_utils.in_hull(points, box_corners) # (N,)
            fg_points = points[fg_pt_flag] # (M, 3)
            cls_label[fg_pt_flag] = labels[k] # (N,)

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(points, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            reg_label[fg_pt_flag, :3] = bboxes[k][:3] - fg_points
            reg_label[fg_pt_flag, 3:] = bboxes[k][3:]

        return cls_label, reg_label
    
    def data_augmentation(self, points, bboxes, mustaug=False):
        """
        :param points: (N, 3)
        :param bboxes: (N, 7)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            points = kitti_utils.rotate_pc_along_y(points, angle)
            bboxes = kitti_utils.rotate_pc_along_y(bboxes, angle)
            bboxes[:, 6] -= angle
            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            points *= scale
            bboxes[:, :6] *= scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            points[:, 2] *= -1.0
            bboxes[:, 2] *= -1.0
            bboxes[:, 6] *= -1.0
            aug_method.append('flip')
        
        bboxes[:, 6][bboxes[:, 6] >  np.pi] -= 2 * np.pi
        bboxes[:, 6][bboxes[:, 6] < -np.pi] += 2 * np.pi
        
        return points, bboxes, aug_method

    def collate_batch(self, sample_list):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert sample_list.__len__() == 1
            return sample_list[0]

        batch_size = sample_list.__len__()
        ans_dict = {}

        for key in sample_list[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for sample in sample_list:
                    max_gt = max(max_gt, sample[key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i, sample in enumerate(sample_list):
                    batch_gt_boxes3d[i, :sample[key].__len__(), :] = sample[key]
                ans_dict[key] = batch_gt_boxes3d
                continue
            
            if key == 'gt_labels':
                max_gt = 0
                for sample in sample_list:
                    max_gt = max(max_gt, sample[key].__len__())
                batch_gt_labels = np.zeros((batch_size, max_gt), dtype=np.int32)
                for i, sample in enumerate(sample_list):
                    batch_gt_labels[i, :sample[key].__len__()] = sample[key]
                ans_dict[key] = batch_gt_labels
                continue

            if isinstance(sample_list[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = sample_list[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([sample[key][np.newaxis, ...] for sample in sample_list], axis=0)

            else:
                ans_dict[key] = [sample[key] for sample in sample_list]
                if isinstance(sample_list[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(sample_list[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict
