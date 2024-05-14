import bpy
import json
import math
import numpy as np
import socket
import os
# from utils.functions import get_circle
# bone = human["shoulder.L"]
from collections import deque
import time, bpy, threading



# a1 = [ -0.029662 m, -0.108663 m, 1.61801 m]
# b1 = [-0.029614 m, -0.111757 m, 1.62512 m]
# c1 = [-0.029607 m, -0.110883 m, 1.62881 m]

def set_eye_brow(lowwer_brow, upper_brow, pc, r):
    # a0 = [ -0.029662 , -0.108663 , 1.61801 ]    # lowwer
    # b0 = [-0.029614 , -0.111757 , 1.62512 ] 
    # c0 = [-0.029607 , -0.110883 , 1.62881 ]     # upper
    # pc , r , trans_upper = get_circle(a0, b0, c0)
    
    z_low = (r*r -(lowwer_brow[1] -pc[1])**2 - (lowwer_brow[0]- pc[0])**2)**0.5 + pc[2]
    z_up =  (r*r -(upper_brow[1] -pc[1])**2 - (upper_brow[0]- pc[0])**2)**0.5 + pc[2]
    return z_low, z_up


def set_angle(human, bone_name, rot):
    # print('set_angel!')
    if len(rot) == 3:
        lastMode = human.pose.bones[bone_name].rotation_mode
        human.pose.bones[bone_name].rotation_mode = 'YXZ'
        human.pose.bones[bone_name].rotation_euler[0] = rot[1]     # Y pitch 
        human.pose.bones[bone_name].rotation_euler[1] = -rot[0]    # yz_form yaw
        human.pose.bones[bone_name].rotation_euler[2] = rot[2]     # roll 
        human.pose.bones[bone_name].rotation_mode = lastMode
    elif len(rot) == 2:
        lastMode = human.pose.bones[bone_name].rotation_mode
        human.pose.bones[bone_name].rotation_mode = 'YXZ'
        human.pose.bones[bone_name].rotation_euler[0] = rot[1]     # Y pitch 
        human.pose.bones[bone_name].rotation_euler[1] = -rot[0]    # yz_form yaw
        # human.pose.bones[bone_name].rotation_euler[2] = rot[2]     # roll 
        human.pose.bones[bone_name].rotation_mode = lastMode
    
def set_location(human, bone_name, loc):
    
    
    human.pose.bones[bone_name].location[0] = loc[0]
    human.pose.bones[bone_name].location[2] = -loc[1]
    # todo: fit depth loc 
    
def set_face_keypoint(human, tem_data, keypoint_loc: dict):
    for k, v in keypoint_loc.items():
        if len(v) == 1:
            set_location(human, k, tem_data[:2, v[0]])
        else:
            set_location(human, k, np.mean(tem_data[:2, v], axis=1))
        

def socket_recv(client_socket, tcp_server_socket):
    recv_data_whole = bytes()
    recv_data = client_socket.recv(1344)  # 56*3*8 = 1344    
    if len(recv_data) == 0 :
        # close socket
        client_socket.close()
        tcp_server_socket.close()
        print('server is close, stop receive!')
        return
        # client_socket, clientAddr = tcp_server_socket.accept() 
    else:
        recv_data_whole += recv_data
        # print('receive data length :', recv_data_whole.__len__())
        # check data length again
        if len(recv_data_whole) == 1344 :   # 56*3*8 = 1344 
            tem_arr = np.frombuffer(recv_data_whole, dtype=np.float64).reshape(3, 56) 
            recv_data_whole = bytes()
        
            # print(tem_arr)

            return tem_arr
    
'''
about 


face_loc =[ 
    8,9,10 jaw/chin
    18-22 left eyebrow 18 19 20 21 22
    23-27 right eyebrow
    28-31  32-36  nose
    37-42 left eye
    43-48 right eye
    
    49-55  56-60  61-68  mouth
]

bone_name = [ 
head chin offset right 4 , None , head chin offset left 4
None (18-19 head eyebrow right 2b), head eyebrow right 2d  ,head eyebrow right 1a,  head eyebrow right 1c, head glabella right
head glabella left,  head eyebrow left 1c, head eyebrow left 1a, head eyebrow left 2d,  None,

nose:  head glabella middle, none ,none ,(head nose tip left , head nose tip right)
eye:  head eyelash upper offset right  , head eyelid lower offset right
index  upper ([1, 37] + [1, 38])/ 2  - ([1,39] +[1,36]) /2  
       lower ([1, 40] + [1, 41])/2  - ([1,39] +[1,36]) /2
       head eyelash upper offset left  , head eyelid lower offset left
       
       head lip corner right 2(49),  head lip upper offset right(50), head lip upper top offset right(51),  ,head lip corner left 2(55)   
       head lip lower bottom right 2(60) , head lip lower bottom right 1(59),head lip corner offset right(61),
       
           
       head lip corner right 1(61),  head lip upper top right adj(62) , head lip upper middle (63) ,head lip upper top left adj(64), head lip corner left 1(65),
        ,head lip lower offset right 1(68,60) , head lip lower offset right 2(mean(67, 59)), head lip lower offset left 2(mean(67,57)), head lip lower offset left 1(66,56)
]

'''

def thread_update(client_socket, tcp_server_socket, bone_dic):
    '''
    tem_out : [pose,  face_loc , left cheek] 
    
    face_loc =[ 
        8,9,10 jaw/chin
        18-22 left eyebrow
        23-27 right eyebrow
        28-31  32-36  nose
        37-42 left eye
        43-48 right eye
        49-55  56-60  61-68  mouth ]
    '''
    while(True):
        tem_ = socket_recv(client_socket, tcp_server_socket)
        
        if len(queue_tem) == 0:
            # pad queue 
            for _ in range(queue_n):
                # todo: load prepare loc
                queue_tem.append(np.empty((3, 56)))
            queue_tem.append(np.empty((3, 56)))
        else:
            queue_tem.append(tem_.copy())
            
        tem_out = np.mean(queue_tem, axis=0)
        tem_pose = tem_out[:, 0].reshape(-1)
        if tem_out[0, -1]:
            tem_out[:, 1:-1] = tem_out[:, 1:-1] / tem_out[0 , -1] * m_scale 
       
        if (len(tem_out) > 1):
        
            # print('begin set angel!')
            # head angle
            set_angle(human, "head_neck_upper", tem_pose)
            
            jaw_loc = np.mean(tem_out[:2, [1,2,3]], axis=1)
            # jaw angle jaw length (-0.037046 m ,-0.116921 m)
            jaw_length = 0.079875
            jaw_angle = [math.asin(i/jaw_length)*0.95 for i in jaw_loc]
            
            # eye lid loc
            
            
            set_angle(human, 'head jaw', jaw_angle)
           
            # set_angle(human, '',)
            
            set_face_keypoint(human, tem_out[:2, 1:-1], bone_dic)
            # set_face_keypoint()
            queue_tem.popleft()
        # time.sleep(0.1) #update rate in seconds
        


if __name__ == '__main__':
    
    # eye lid lash
    # aL = [ -0.029662 , -0.108663 , 1.61801 ]    # lowwer
    # bL = [-0.029614 , -0.111757 , 1.62512 ] 
    # cL = [-0.029607 , -0.110883 , 1.62881 ]     # upper
    # L_pc , L_r ,L_upper = get_circle(aL, bL, cL, trnasfer=True)
    
    # aR = [0.031787 , -0.109222, 1.6184]
    # bR = [0.032148 , -0.112595, 1.62537]
    # cR = [0.032025 , -0.111889, 1.6291]
    # R_pc ,R_r, R_upper = get_circle(aR, bR, cR)
   
    # old version
    # bpy.context.scene.objects.active = human
    # bones name to index
    bone_dic = {    
                        # 'head chin offset right 4': [0], #  8 - 7- 1(pose out)
                        'head jaw': [1, 2, 3],
                        
                        'head eyebrow right 2b': [4], # 18，19 - 14-1
                        'head eyebrow right 1a': [5],
                        'head eyebrow right 1c': [6], 
                        'head glabella right'  : [7],
                        'head glabella left'   : [8],
                        'head eyebrow left 1c': [9],
                        'head eyebrow left 1a': [10], 
                        'head eyebrow left 2d': [11],
                        
                        #eye lid
                        
                        'head eyelid corner inner right 1': [25], #40-14-1
                        'head eyelash upper offset right': [23, 24], #38，39
                        'head eyelid lower offset right' : [26, 27], #41，42
                        'head eyelid corner inner left 1': [28],  #43 -14-1
                        'head eyelash upper offset right': [29, 30],  #44，45
                        'head eyelid lower offset right' : [32, 33],  #47，48
                        
                        # mouth
                        'head mouth upper offset right' : [34],  # 49-14-1
                        # 'head lip upper offset right' : [35], 
                        # 'head lip upper top offset right' : [36], 
                       
                        'head mouth upper offset left' : [40], # 55 -14-1   
                        # 'head lip lower bottom right 2': [45] , 
                        # 'head lip lower bottom right 1':[44],
                        
                        'head lip corner offset right': [46],
                        'head lip corner offset left':[50],# 65-14-1
                        
                        # 'head lip corner right 1': [46], # 61 -14-1  
                        # 'head lip upper top right adj':[47] , #62-14-1
                        # 'head_lip_upper_middle': [48] ,#
                        # 'head lip upper top left adj': [49], 
                        
                        # 'head lip lower offset right 1':[45, 53], #60 ,68 -14
                        # 'head lip lower offset right 2':[44, 52], #59 ,67
                        # 'head lip lower offset left 2':[42, 52], #57, 67-14-1
                        # 'head lip lower offset left 1':[41, 51] # 56, 66
            }

    human = bpy.data.objects['Armature']
    bpy.context.view_layer.objects.active = human
    bpy.ops.object.mode_set(mode='POSE')
    # set head length as scale
    m_scale = human.pose.bones['head_neck_upper'].length
 
    # tcp socket
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM )
   
    hostname = 'BJL'
    port =  8800
    address = (hostname, port)
    # bind
    tcp_server_socket.bind(address)
    tcp_server_socket.listen(5)
    # clientAddr  your blind (hostname | ip ，port)
    client_socket, clientAddr = tcp_server_socket.accept()
    recv_data_whole = bytes()
    
    queue_tem = deque()
    # the queue length: n+1 
    queue_n = 1
    #test 
    # tem_ = np.arange(3*56).reshape(3, 56)
    thread = threading.Thread(target=thread_update, args=(client_socket, tcp_server_socket, bone_dic))
    thread.start()
