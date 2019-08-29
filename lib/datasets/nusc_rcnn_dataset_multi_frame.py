import os
import random
import numpy as np
import torch
from pyquaternion import Quaternion
from lib.datasets.nusc_dataset import nuScenesDataset
import lib.utils.kitti_utils as kitti_utils
from lib.config import cfg

class nuScenesRCNNDataset_mf(nuScenesDataset):

    def __init__(self, nusc, split, mode, subset=False, subset_file=None,
                 subset_fold=4, classes='all', npoints=16384, random_select=True, 
                 logger=None):
        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        super(nuScenesRCNNDataset_mf, self).__init__(nusc=nusc, split=split)
        
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

        if subset and subset_file is not None:
            self.logger.info('Loading a %s subset from %s ...' % (self.mode, subset_file))
            with open(subset_file, 'r') as f:
                self.sample_tokens = [token.rstrip() for token in f.readlines()]
            self.logger.info('Done, load a %s subset with %d samples.' % (self.mode, len(self.sample_tokens)))
                
        self.preprocess_rpn_training_data()
        
        if subset and subset_file is None:
            self.logger.info('Sampling a %s subset ...' % self.mode)
            subset_length = int(self.sample_tokens.__len__() / subset_fold)
            self.sample_tokens = random.sample(self.sample_tokens, subset_length)
            self.logger.info('Done, sample a %s subset with %d samples.' % (self.mode, len(self.sample_tokens)))
    
    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_token is stored in self.sample_tokens
        """
        self.logger.info('Filtering %s samples ... ' % self.mode)
        valid_tokens = []
        for sample_token in self.sample_tokens:
            sample_info = self.nusc.get('sample', sample_token)
            if sample_info['prev'] != '':
                if self.mode == 'TRAIN':
                    sample_ann_infos = self.filterate_anns(self.get_anns(sample_info), add_label=False)
                    if sample_ann_infos.__len__() > 0:
                        valid_tokens.append(sample_token)
                else:
                    valid_tokens.append(sample_token)

        self.logger.info('Done: filter valid %s samples: %d / %d' % (self.mode, len(valid_tokens), len(self.sample_tokens)))
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

    def filterate_anns(self, ann_infos, min_pts=5, add_label=True):
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
    
    def point_sampling(self, sample_lidar_points, sample_lidar_intensity):
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
        return sample_lidar_points, sample_lidar_intensity
    
    @staticmethod
    def prepare_pts_outputs(sample_outputs, sample_lidar_points, sample_lidar_points_prev,
                            sample_lidar_intensity, sample_lidar_intensity_prev, shuf=None):
        sample_lidar_frame = np.concatenate((np.ones(sample_lidar_points.__len__()), 
                                             np.zeros(sample_lidar_points_prev.__len__())))   
        sample_lidar_points = np.concatenate((sample_lidar_points, sample_lidar_points_prev))
        sample_lidar_intensity = np.concatenate((sample_lidar_intensity, sample_lidar_intensity_prev))
        if shuf is None:
            shuf = np.arange(sample_lidar_points.__len__())
            np.random.shuffle(shuf)
        sample_lidar_points = sample_lidar_points[shuf]
        sample_lidar_intensity = sample_lidar_intensity[shuf]
        sample_lidar_frame = sample_lidar_frame[shuf]

        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((sample_lidar_points, sample_lidar_intensity), axis=1)  # (N, C)
        else:
            pts_input = sample_lidar_points

        sample_outputs['pts_input'] = pts_input.astype(np.float32)
        sample_outputs['pts_rect'] = sample_lidar_points.astype(np.float32)
        sample_outputs['pts_features'] = sample_lidar_intensity.astype(np.float32)
        sample_outputs['pts_frame'] = sample_lidar_frame.astype(np.int32)

        return sample_outputs
    
    @staticmethod
    def prepare_rpn_labels(sample_outputs, rpn_cls_label, rpn_cls_label_prev,
                           rpn_reg_label, rpn_reg_label_prev, shuf=None):
        rpn_cls_label = np.concatenate((rpn_cls_label, rpn_cls_label_prev))
        rpn_reg_label = np.concatenate((rpn_reg_label, rpn_reg_label_prev))
        if shuf is None:
            shuf = np.arange(rpn_cls_label.__len__())
            np.random.shuffle(shuf)
        rpn_cls_label = rpn_cls_label[shuf]
        rpn_reg_label = rpn_reg_label[shuf]

        sample_outputs['rpn_cls_label'] = rpn_cls_label.astype(np.float32)
        sample_outputs['rpn_reg_label'] = rpn_reg_label.astype(np.float32)

        return sample_outputs
    
    @staticmethod
    def prepare_gt(sample_outputs, sample_ann_bboxes, sample_ann_bboxes_prev,
                   sample_ann_labels, sample_ann_labels_prev):
        gt_frame = np.concatenate((np.ones(sample_ann_bboxes.__len__()), np.zeros(sample_ann_bboxes_prev.__len__())))
        sample_ann_bboxes = np.concatenate((sample_ann_bboxes, sample_ann_bboxes_prev))
        sample_ann_labels = np.concatenate((sample_ann_labels, sample_ann_labels_prev))
        
        shuf = np.arange(sample_ann_bboxes.__len__())
        np.random.shuffle(shuf)
        sample_ann_bboxes = sample_ann_bboxes[shuf]
        sample_ann_labels = sample_ann_labels[shuf]
        gt_frame = gt_frame[shuf]

        sample_outputs['gt_boxes3d'] = sample_ann_bboxes.astype(np.float32)
        sample_outputs['gt_label'] = sample_ann_labels.astype(np.int32)
        sample_outputs['gt_frame'] = gt_frame.astype(np.int32)

        return sample_outputs
    
    def get_rpn_sample(self, sample_info):
        
        # get original data and preprocess point cloud and labels
        sample_outputs = {'token': sample_info['token'], 'random_select': self.random_select}
        
        # get first frame
        sample_lidar_pc, sample_lidar_ep_info, sample_lidar_cs_info = self.get_lidar(sample_info)
        sample_lidar_pc = self.remove_useless_points(sample_lidar_pc)

        # get prev frame
        sample_info_prev = self.nusc.get('sample', sample_info['prev'])
        sample_lidar_pc_prev, sample_lidar_ep_info_prev, sample_lidar_cs_info_prev = self.get_lidar(sample_info_prev)
        sample_lidar_pc_prev = self.remove_useless_points(sample_lidar_pc_prev)

        # point cloud coordinate transform for better usage of labels
        sample_lidar_points, sample_lidar_intensity = self.pc_trans(sample_lidar_pc, sample_lidar_ep_info, sample_lidar_cs_info)
        sample_lidar_points_prev, sample_lidar_intensity_prev = self.pc_trans(sample_lidar_pc_prev, sample_lidar_ep_info_prev, sample_lidar_cs_info_prev)

        # points sampling
        sample_lidar_points, sample_lidar_intensity = self.point_sampling(sample_lidar_points, sample_lidar_intensity)
        sample_lidar_points_prev, sample_lidar_intensity_prev = self.point_sampling(sample_lidar_points_prev, sample_lidar_intensity_prev)
        
        # translate prev to current coordinate system
        shift = np.array(sample_lidar_ep_info_prev['translation']) - np.array(sample_lidar_ep_info['translation'])
        sample_lidar_points_prev += shift

        # change to KITTI axis
        sample_lidar_points = sample_lidar_points[:, [0, 2, 1]]
        sample_lidar_points[:, 1] *= -1.0
        sample_lidar_points_prev = sample_lidar_points_prev[:, [0, 2, 1]]
        sample_lidar_points_prev[:, 1] *= -1.0
        
        if self.mode == 'TEST':
            sample_outputs = self.prepare_pts_outputs(sample_outputs, sample_lidar_points, sample_lidar_points_prev,
                                                      sample_lidar_intensity, sample_lidar_intensity_prev)
            return sample_outputs
        
        # get anns if not TEST
        sample_ann_infos = self.filterate_anns(self.get_anns(sample_info))
        sample_ann_infos_prev = self.filterate_anns(self.get_anns(sample_info_prev))

        # annotation translate
        sample_ann_infos = self.ann_trans(sample_ann_infos, sample_lidar_ep_info)
        sample_ann_infos_prev = self.ann_trans(sample_ann_infos_prev, sample_lidar_ep_info)

        # gether annotations
        sample_ann_bboxes, sample_ann_labels = self.anns_to_bboxes(sample_ann_infos)
        sample_ann_bboxes_prev, sample_ann_labels_prev = self.anns_to_bboxes(sample_ann_infos_prev)

        # data augmentation, points and bboxes changed
        if cfg.AUG_DATA:
            [sample_lidar_points, sample_lidar_points_prev], [sample_ann_bboxes, sample_ann_bboxes_prev], aug_method = \
                self.data_augmentation([sample_lidar_points, sample_lidar_points_prev], [sample_ann_bboxes, sample_ann_bboxes_prev])
            sample_outputs['aug_method'] = aug_method
        
        # merge frames
        shuf = np.arange(sample_lidar_points.__len__() + sample_lidar_points_prev.__len__())
        np.random.shuffle(shuf)
        
        if not cfg.RPN.FIXED:
            rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(sample_lidar_points, sample_ann_bboxes, sample_ann_labels)
            rpn_cls_label_prev, rpn_reg_label_prev = self.generate_rpn_training_labels(sample_lidar_points_prev, sample_ann_bboxes_prev, sample_ann_labels_prev)
            sample_outputs = self.prepare_rpn_labels(sample_outputs, rpn_cls_label, rpn_cls_label_prev,
                                                     rpn_reg_label, rpn_reg_label_prev, shuf=shuf)

        sample_outputs = self.prepare_pts_outputs(sample_outputs, sample_lidar_points, sample_lidar_points_prev,
                                                  sample_lidar_intensity, sample_lidar_intensity_prev, shuf=shuf)
        sample_outputs = self.prepare_gt(sample_outputs, sample_ann_bboxes, sample_ann_bboxes_prev,
                                         sample_ann_labels, sample_ann_labels_prev)

        return sample_outputs
    
    @staticmethod
    def generate_rpn_training_labels(points, bboxes, labels):
        cls_label = np.zeros((points.__len__()))
        reg_label = np.zeros((points.__len__(), 7))
        shrink_bboxes = kitti_utils.remove_ground_box3d(bboxes, ground_height=0.05)
        shrink_corners = kitti_utils.boxes3d_to_corners3d(shrink_bboxes)
        #extend_bboxes = kitti_utils.enlarge_box3d(bboxes, extra_width=0.05)
        #extend_corners = kitti_utils.boxes3d_to_corners3d(extend_bboxes)
        for k in range(bboxes.__len__()):
            box_corners = shrink_corners[k]
            fg_pt_flag = kitti_utils.in_hull(points, box_corners) # (N,)
            fg_points = points[fg_pt_flag] # (M, 3)
            cls_label[fg_pt_flag] = labels[k] # (N,)

            # enlarge the bbox3d, ignore nearby points
            #extend_box_corners = extend_corners[k]
            #fg_enlarge_flag = kitti_utils.in_hull(points, extend_box_corners)
            #ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            #cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = bboxes[k][:3].copy()  # (x, y, z)
            center3d[1] -= bboxes[k][3] / 2
            reg_label[fg_pt_flag, :3] = center3d - fg_points  # Now y is the true center of 3d box 20180928

            # size and angle encoding
            reg_label[fg_pt_flag, 3:] = bboxes[k][3:]

        return cls_label, reg_label
    
    def data_augmentation(self, points, bboxes, mustaug=False):
        """
        :param points: (N, 3)
        :param bboxes: (N, 7)
        :return:
        """

        one_frame = False
        if not isinstance(points, list):
            points = [points]
            bboxes = [bboxes]
            one_frame = True
        num_frame = len(points)

        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            for i in range(num_frame):
                points[i] = kitti_utils.rotate_pc_along_y(points[i], angle)
                bboxes[i] = kitti_utils.rotate_pc_along_y(bboxes[i], angle)
                bboxes[i][:, 6] -= angle
            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            for i in range(num_frame):
                points[i] *= scale
                bboxes[i][:, :6] *= scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            for i in range(num_frame):
                points[i][:, 2] *= -1.0
                bboxes[i][:, 2] *= -1.0
                bboxes[i][:, 6] *= -1.0
            aug_method.append('flip')
        
        for i in range(num_frame):
            bboxes[i][:, 6][bboxes[i][:, 6] >  np.pi] -= 2 * np.pi
            bboxes[i][:, 6][bboxes[i][:, 6] < -np.pi] += 2 * np.pi
        
        if one_frame:
            points = points[0]
            bboxes = bboxes[0]
        
        return points, bboxes, aug_method

    def collate_batch(self, sample_list):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert sample_list.__len__() == 1
            return sample_list[0]

        batch_size = sample_list.__len__()
        ans_dict = {}
        
        max_gt = -1
        for key in sample_list[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                if max_gt == -1:
                    for sample in sample_list:
                        max_gt = max(max_gt, sample[key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i, sample in enumerate(sample_list):
                    batch_gt_boxes3d[i, :sample[key].__len__(), :] = sample[key]
                ans_dict[key] = batch_gt_boxes3d
                continue
            
            if key == 'gt_label' or key == 'gt_frame':
                if max_gt == -1:
                    for sample in sample_list:
                        max_gt = max(max_gt, sample[key].__len__())
                batch_gt_label = np.zeros((batch_size, max_gt), dtype=np.int32)
                for i, sample in enumerate(sample_list):
                    batch_gt_label[i, :sample[key].__len__()] = sample[key]
                ans_dict[key] = batch_gt_label
                continue

            if isinstance(sample_list[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = sample_list[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([sample[key][np.newaxis, ...] for sample in sample_list])

            else:
                ans_dict[key] = [sample[key] for sample in sample_list]
                if isinstance(sample_list[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(sample_list[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict
