# Flame drive blender smpl face model 
<p align="left">
<a href="Doc/README_CN.md"><img src="https://img.shields.io/badge/文档-中文版-blue.svg" alt="CN doc"></a> 
<a href="README.md"><img src="https://img.shields.io/badge/document-English-blue.svg" alt="EN doc"></a>
</p>


  <p align="center"> 
  <img src="cut.gif">
  </p>
  <p align="center">脸部识别驱动脸部模型的表情和姿势 <p align="center">


这是一个blender插件，用于驱动smplx的脸部模型。它有两个部分：人脸识别（使用FLAME）和人脸模型驱动（blender smplx插件）。

## 项目描述
blender插件部分基于[blender-smplx-addon](https://smpl-x.is.tue.mpg.de/)，人脸识别部分基于[DECA](https://github.com/yfeng95/DECA)

  
## 项目流程
先运行blender插件，再运行本地的人脸识别部分


### blender插件部分

#### 下载插件

* 从[这里](https://drive.google.com/file/d/1QYBQjPlzC7Xk06JVYWWQSltdx7gg70c_/view?usp=sharing)下载插件，或者从release页面下载


#### 安装插件

* 在blender中选择`File` -> `User Preferences`
  <p align="center">   
  <img src="load_addon.gif">
  </p>


* 选择`Add-ons` -> `Install...`, 选择插件文件

    
### 脸部识别部分

克隆仓库文件:
  ```bash
  git clone https://github.com/Brusselslee/blender_smplx_face
  ```  

#### 本地环境要求
 
* 脸部算法（可选，用于检测脸部，我使用yolov8预训练模型）  
  
* 安装依赖
  ```bash
  pip install -r requirements.txt
  ```
* 准备数据 
  
  下载预训练模型，从[这里](https://drive.google.com/file/d/16bfjajmNJtsQT6pXN0me_FKFGJnhCTPY/view?usp=sharing)下载，将文件`flame_encode_params.pkl`放在`data`文件夹下。

  yolov8脸部预训练模型来自[这里](https://drive.google.com/file/d/1SxyTynMZhRJu0goGhNL4PVUFTMgGEgv4/view?usp=sharing)，将文件`yolov8n_100e.pt`放在`data`文件夹下。


## 快速开始

* 在blender中选择`View` -> `Sidebar`

* 在侧栏中选择`Webcam`


* 在界面中选择modle模型，添加一个模型实例
  

* 选择模型mesh部分，然后点击start按钮
  <p align="center">   
  <img src="start.gif">
  </p>

* 运行脸部识别部分
  切换到smpl_face_blender文件夹下 运行
  ```bash
  python demo_video.py -i 0
  ```

* 更多教程视频可以看[SMPL-X Blender Add-On -- Tutorial](https://www.youtube.com/watch?v=DY2k29Jef94) 
* 我的插件[视频](https://www.bilibili.com/video/BV12D421A7Gf/?spm_id_from=333.999.0.0&vd_source=68a5de8a8ffee0752ac60feb59fc0d68)


## 未来计划
- [ ] 对指定的脸进行追踪(多脸情况下的追踪)
- [ ] 添加更多的blender addon用户交互
- [ ] 扩展smplx模型的表情
- [ ] 驱动自定义模型
- [ ] 合并两个部分（blender addon和face recognition）的版本，在blender里安装pytorch？

## 感谢
程序和脚本都是基于外部资源，我们在此特别感谢：  
- [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) 和 [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME) 用于FLAME模型  
- [blender](https://www.blender.org/download/) 
- [smplx](https://smpl-x.is.tue.mpg.de/) 用于blender模型
- [ultralytics](https://github.com/ultralytics/ultralytics) 用于人脸检测和识别
- [face_landmarks](https://github.com/1adrianb/face-alignment) 用于人脸关键点
