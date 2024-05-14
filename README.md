# Flame drive blender smpl face model 

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

* download the addon from [here](https://drive.google.com/file/d/1QYBQjPlzC7Xk06JVYWWQSltdx7gg70c_/view?usp=sharing)


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



## quick start

* in blender choose the smplx model to addon
  

* push the webcam botton
  <p align="center">   
  <img src="Doc/start.gif">
  </p> 

* run the face recognition part
  cd to the smpl_face_blender 
  ```bash
  python demo_video.py -i 0
  ```



* more video can watch [SMPL-X Blender Add-On -- Tutorial](https://www.youtube.com/watch?v=DY2k29Jef94) 
* my addon [video]() 



## Usage

* in blender choose the smplx model to addon
  

* push the webcam botton
  <p align="center">   
  <img src="Doc/start.gif">
  </p>


## TODO
- [ ] more blender addon user interaction
- [ ] extend smplx model expression
- [ ] drive custom model




## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/YadiraF/DECA/blob/master/LICENSE) file.
By downloading and using the code and model you agree to the terms in the [LICENSE](https://github.com/YadiraF/DECA/blob/master/LICENSE). 

## Acknowledgements
For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.  
Here are some great resources we benefit:  
- [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) and [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME) for the FLAME model  
- [blender](https://www.blender.org/download/) 
- [smplx](https://smpl-x.is.tue.mpg.de/) for blender model
- [ultralytics](https://github.com/ultralytics/ultralytics) for face dectection and recognition
- [face_landmarks](https://github.com/1adrianb/face-alignment) for face landmarks
- [face_segmentation](https://github.com/YuvalNirkin/face_segmentation) for skin mask


