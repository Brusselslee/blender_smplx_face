# Flame drive blender smpl face model 
<p align="left">
<a href="Doc/README_CN.md"><img src="https://img.shields.io/badge/文档-中文版-blue.svg" alt="CN doc"></a> 
<a href="README.md"><img src="https://img.shields.io/badge/document-English-blue.svg" alt="EN doc"></a>
</p>

  <p align="center"> 
  <img src="Doc/cut.gif">
  </p>
  <p align="center">face recognition to drive blender model to emotion and pose <p align="center">


This is a blender addon for driving the face model of smplx.It has two parts: face recognition(use FLAME) and face model driving(blender smplx addon).


## Description
The blender addon part base on [blender-smplx-addon](https://smpl-x.is.tue.mpg.de/) download and face recognition part base on [DECA](https://github.com/yfeng95/DECA)


  
## Getting Started
run blender addon frist, then local face recognition part
### blender addon part

#### download the addon

* download the addon from [here](https://drive.google.com/file/d/1QYBQjPlzC7Xk06JVYWWQSltdx7gg70c_/view?usp=sharing), or download from release page


#### blender load
* in blender load the addon

    <p align="center">   
    <img src="Doc/load_addon.gif">
    </p>



### face recognition part

Clone the repo:
  ```bash
  git clone https://github.com/YadiraF/DECA
  ```  

#### Requirements
 
* face-alignment (Optional for detecting face, i use yolov8 pretrained)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
 

* Prepare data   
  
  download encoded pretrained model from [here](https://drive.google.com/file/d/16bfjajmNJtsQT6pXN0me_FKFGJnhCTPY/view?usp=sharing), put the file `flame_encode_params.pkl` under the `data`.

  yolov8 face pretrained model from [here](https://drive.google.com/file/d/1SxyTynMZhRJu0goGhNL4PVUFTMgGEgv4/view?usp=sharing), put the file `yolov8n_100e.pt` under the `data`.


* run the face recognition part
  cd to the DECA folder
  ```bash
  python demo_video.py -i 0
  ```
    


## quick start


* in blender make sure addon was loaded , and select `View` -> `Sidebar`, in Sidebar select `Webcam`

* in blender choose the smplx model to add 
  

* push the webcam `start` botton
  <p align="center">   
  <img src="Doc/start.gif">
  </p>

* run the face recognition part
  cd to the smpl_face_blender 
  ```bash
  python demo_video.py -i 0
  ```

* more video can watch [SMPL-X Blender Add-On -- Tutorial](https://www.youtube.com/watch?v=DY2k29Jef94) 
* my addon [video](https://www.bilibili.com/video/BV12D421A7Gf/?spm_id_from=333.999.0.0&vd_source=68a5de8a8ffee0752ac60feb59fc0d68) 


## TODO
- [ ] tracking on target face (multi-face situation tracking)
- [ ] more blender addon user interaction
- [ ] extend smplx model expression
- [ ] drive custom model
- [ ] merge two part(blender addon and face recognition) version,in blender install pytorch ?


## Acknowledgements
For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.  
Here are some great resources we benefit:  
- [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) and [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME) for the FLAME model  
- [blender](https://www.blender.org/download/) 
- [smplx](https://smpl-x.is.tue.mpg.de/) for blender model
- [ultralytics](https://github.com/ultralytics/ultralytics) for face dectection and recognition
- [face_landmarks](https://github.com/1adrianb/face-alignment) for face landmarks



