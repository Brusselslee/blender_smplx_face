# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import numpy as np
import torch

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'

class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None,...])
        if out[0][0] is None:
            return [0]
        else:
            bbox = out[0][0].squeeze()
            return bbox, 'bbox'

class YOLOV8(object):
    def __init__(self, device = 'gpu'):
        import torch
        import torch.backends.cudnn as cudnn
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'mps' if torch.backends.mps.is_available() else device
        
        if device == 'cuda': cudnn.benchmark = True
        
        from ultralytics import YOLO
        self.model = YOLO('./yolov8n_100e.pt')
        self.model.to(device)
        
        self.box_format = 'bbox'
        
        
    def run(self, img):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        
        # conf = 0.5
        results = self.model.predict(img, stream=False, verbose=False)
        img_res = results[0]
        image_boxes = img_res.boxes.xyxy.cpu().numpy()
        # image_scores = img_res.boxes.conf.cpu().numpy()
        filtered_boxes = []
        if image_boxes.shape[0] == 0:
            return filtered_boxes, self.box_format
        
        # multiple face tracking
        
        
        img_w, img_h = img.shape[1], img.shape[0]
        idx = self.filter_center_box(image_boxes, img_w, img_h)    
        
        center_box = image_boxes[idx]
            
        filtered_boxes = center_box
            
        
                
        return filtered_boxes, self.box_format

    
    def filter_center_box(self, boxes, img_x, img_y):
        
        distance = [img_x, img_y]
        idx = 0
        for i, k in enumerate(boxes):
            # just count x_axis
            if abs(k[0]- img_x) < distance[0] :
                distance[0] = abs(k[0]- img_x)
                idx = i
                
        return idx
        
        