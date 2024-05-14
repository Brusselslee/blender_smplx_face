
# import trimesh
 
# # 读取PLY文件
# mesh = trimesh.load('model2019_fullHead_lvl2.ply')
 
# # 获取顶点坐标
# vertices = mesh.vertices

# # face
# face = mesh.faces


# #法线（Normals）
# normals = mesh.vertex_normals


# is_watertight = mesh.is_watertight

# mesh_smooth = mesh.smoothed() # 平滑
# mesh_subdivide = mesh.subdivide() # 细分 

# mesh.show()

model_path ='./flame_encode_params.pkl'

import torch
import pickle

# Load the pickled file using pickle.load
with open(model_path, 'rb') as f:
    data = pickle.load(f)

# Convert the loaded data to a PyTorch tensor
tensor_data = torch.tensor(data)

# Example: Print the loaded tensor
print(tensor_data)