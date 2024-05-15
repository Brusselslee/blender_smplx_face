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

from collections import deque
import socket
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points



def send_tcp(tcp_client_socket, message: np.array):
    
    tcp_client_socket.send(message.tobytes())
    # print(f'----- sending message ! -----')
    # recv_data = tcp_client_socket.recv(1024)
    # print('receive the postback :', recv_data.decode('gbk'))
    
    

def main(args):
    # # tcp connection
    tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM )
    server_ip = socket.gethostname()
    server_port = 8800
    tcp_client_socket.connect((server_ip, server_port))
    
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    
    streamdata = datasets.WebcamStream(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
    # streamdata = datasets.VideoData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
    
    # testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    # 3 to 5 enough
    n_pre = 3
    n_next = 0
    buffer_len = n_pre+ n_next    
    face_buffer = deque()
    # for i in range(len(testdata)):
    
    for img, trans, face in streamdata:
        with torch.no_grad():
            images = img.to(device)[None,...]
            codedict = deca.encode(images)
            
            shape = codedict['shape'].cpu().numpy().copy()
            pose = codedict['pose'].cpu().numpy().copy()
            exp = codedict['exp'][:,:10].cpu().numpy().copy()
             
            # face_ = 0 if not face else 1
            # face_arr = np.array([face_]).reshape(1,1)
            # whether detect face
            if face:
                tem_data= np.hstack([shape, exp, pose]) # type = np.float32
            else:
                tem_data= np.zeros((1,116),dtype=np.float32) # type = np.float32
            
            face_buffer.append(tem_data.copy())    
            
            if len(face_buffer) < buffer_len:
                face_buffer.append(tem_data.copy())
                
            else:
                tem_ave = np.mean(np.array(face_buffer),axis=0)
                
                tem_data = tem_ave.copy()
                face_buffer.popleft()
            
            # socket connection
            send_tcp(tcp_client_socket, tem_data)
    
    
    
    print('----- done ! -----')
    # print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flame_blender:  Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', 
                        default='./jimcarrey_cut.mp4',        #'TestSamples/examples/', 
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='yolov8', type=str,
                        choices=['fan', 'mtcnn', 'yolov8'],
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())