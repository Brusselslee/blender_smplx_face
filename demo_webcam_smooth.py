
import argparse
import json
import socket
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
from utils.pose import matrix2angle, P2sRt
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark

from utils.tddfa_util import str2bool
import torch.backends.cudnn as cudnn
import torch
from ultralytics import YOLO

from detection_yolo import detect_faces
from PIL import Image
import time
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    _, R, _ = P2sRt(P)
    pose = matrix2angle(R)
    return P, np.array(pose)
    

def get_model(weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    
    if device == 'cuda': cudnn.benchmark = True
    
    model = YOLO(weights)                
    model.to(device)
    return model


def send_tcp(tcp_client_socket, message: np.array):
    
    tcp_client_socket.send(message.tobytes())
    # print(f'----- sending message ! -----')
    # recv_data = tcp_client_socket.recv(1024)
    # print('receive the postback :', recv_data.decode('gbk'))

    
def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    
    
    if args.send == 'send':
        # load base model
        base_ = np.load('base.npz')
        base_out = base_['arr_0']
        base_pose = base_['arr_1']
        # pose_angle , face point , first point is left cheek as the scale 
        base_orig = np.hstack((base_pose.reshape(3, 1), base_out[:, 7:10], base_out[:, 17:], base_out[:, :1]))
        
        #
        tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM )
        # socket info
        server_ip = 'BJL'
        server_port = 8800
        # 
        tcp_client_socket.connect((server_ip, server_port))
    
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX
        if args.detector == 'yolov8' or args.detector == 'yolov8-pose':
            yolo_box = get_model(args.weight)
        else:
            face_boxes = FaceBoxes_ONNX()
            # face_boxes = FaceBoxes_ONNX()
        # gpu_mode = args.mode == 'gpu'
        # tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)    
        tddfa = TDDFA_ONNX(**cfg)
        
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    
        if args.detector == 'yolov8' or args.detector == 'yolov8-pose':
            yolo_box = get_model(args.weight)
        else:
            face_boxes = FaceBoxes_ONNX()
        

    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    reader = imageio.get_reader("<video0>")

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver1 = deque()
    queue_ver = deque()
    queue_frame = deque()

    # run
    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = []
            # pose_key = []
            # just select single person
            if args.detector == 'yolov8':
                boxes = detect_faces(yolo_box ,[Image.fromarray(frame_bgr)], box_format='xywh',th=0.38)
                boxes = [boxes[0]]
            # elif args.detector == 'yolov8-pose':
            #     results = yolo_box(frame_bgr)
            
            #     boby_boxes = results[0].boxes
            #     key_land = results[0].keypoints
            #
            #     boby_boxes = [key_land[0,4] ,boby_boxes.xyxy[0,1], ]
                
            else:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                
            # boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes, args.detector)
            
          
            # todo: just transfor essential keypoint
            ver = tddfa.recon_stand(param_lst, roi_box_lst)[0]
            ver1 = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            
            
            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver1], args.detector, crop_policy='landmark')
            ver = tddfa.recon_stand(param_lst, roi_box_lst)[0]
            ver1 = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            
            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
                queue_ver1.append(ver1.copy())
            queue_ver.append(ver.copy())
            queue_ver1.append(ver1.copy())
            
            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], args.detector, crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes, args.detector)

            ver = tddfa.recon_stand(param_lst, roi_box_lst)[0]
            ver1 = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_ver1.append(ver1.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver1  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            
            ver_ave1 = np.mean(queue_ver1, axis=0)
            if args.send == 'send':
                # base loc
                ver_ave = np.mean(queue_ver, axis=0)
                # 
                r, pose = calc_pose(param_lst[0])
                
                ver_lo = ver_ave.T
                center1 = (ver_lo[0] + ver_lo[16]) / 2
                
                out_arr = []
                if max(abs(i) for i in pose) <= 65 * np.pi / 180:
                
                    out_x = ver_ave[0] - center1[0]
                    out_y = ver_ave[1] - center1[1]
                    
                    out_arr = np.array((out_x, out_y, ver_ave[2]))
                    tem_data = np.hstack((pose.reshape(3,1), out_arr[:, 7:10] ,out_arr[:, 17:], out_arr[:, :1]))
                else:
                    # todo: set limit
                    tem_data = np.hstack((pose.reshape(3,1), np.empty((3, 54), dtype=np.float64), out_arr[:, :1]))
                    # out_arr = [0]
                
                
                # base_out [3, 68]  base_pose.shape [1, 3]
                tem_data[:, :1] = tem_data[: , :1] - base_orig[: , :1]
                # todo: depth computing
                tem_data[:2, 1:-1] = base_orig[:2, 1:-1] / base_orig[0, -1] * tem_data[0 , -1] - tem_data[:2, 1:-1]  
                
                tem_data[2:,1:4] = base_orig[2:,1:4] / base_orig[0, -1] * tem_data[0 , -1] - tem_data[2: ,1:4] 
                '''
                face_loc =[ 
                8,9,10 jaw/chin
                18-22 left eyebrow
                23-27 right eyebrow
                28-31  32-36  nose
                37-42 left eye
                43-48 right eye
                49-55  56-60  61-68  mouth ]
            
                '''
                send_tcp(tcp_client_socket, tem_data)
                # tcp_client_socket.connect((server_ip, server_port))
                # tcp_client_socket.send(tem_data.tobytes())
          
            if args.opt == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave1)  # since we use padding
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave1, size=1)
            elif args.opt == '3d':
                img_draw = render(queue_frame[n_pre], [ver_ave1], tddfa.tri, alpha=0.7)
            else:
                raise ValueError(f'Unknown opt {args.opt}')
            cv2.imshow('image', img_draw)
            
            k = cv2.waitKey(20)
            if (k & 0xff == ord('q')):
                break

            queue_ver.popleft()
            queue_ver1.popleft()
            queue_frame.popleft()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-w', '--weight', type=str, default='yolov8n_100e.pt')
    parser.add_argument('-detector', default='yolov8', type=str, help='the detector',choices=['', 'yolov8','yolov8-pose'])
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-s', '--send', default='send', type=str, help='weather send message', choices=['send', ''])
    parser.add_argument('-o', '--opt', type=str, default='3d', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=True)
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    args = parser.parse_args()
    main(args)
