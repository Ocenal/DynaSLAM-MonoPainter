# DynaSLAM_MonoPainter
A supplementary for DynaSLAM

## Getting Started
- Install ORB-SLAM2 prerequisites: C++11 or C++0x Compiler, Pangolin, OpenCV and Eigen3  (https://github.com/raulmur/ORB_SLAM2).
- Install boost libraries with the command `sudo apt-get install libboost-all-dev`.
- Install python 2.7, keras and tensorflow, and download the `mask_rcnn_coco.h5` model from this GitHub repository: https://github.com/matterport/Mask_RCNN/releases. 
- Build the target
```
cd DynaSLAM_MonoPainter
chmod +x build.sh
./build.sh
```
- Place the `mask_rcnn_coco.h5` model in the folder `DynaSLAM_MonoPainter/src/python/`.

## Example on TUM Dataset
- Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

- Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. By providing the last and second argument `PATH_TO_MASKS`, dynamic objects are detected with Mask R-CNN. The last argument `PATH_TO_OUTPUT`, specify the folder where the painted frames store.
- A sequence with strong dynamic of len is recommended.
```
./Examples/MonoPainter/mono_painter Vocabulary/ORBvoc.txt Examples/MonoPainter/TUMX.yaml PATH_TO_SEQUENCE_FOLDER PATH_TO_MASKS PATH_TO_OUTPUT
```

## Example of painted frame
(data from TUM-dataset/rgbd_dataset_freiburg3_walking_xyz)
![o1341846320 699927](https://github.com/Ocenal/DynaSLAM-MonoPainter/assets/61320052/29cc216d-70dc-40a9-a51a-eafcc201b014)
![1341846320 699927](https://github.com/Ocenal/DynaSLAM-MonoPainter/assets/61320052/584f9597-fd02-475c-8d71-57eb7553393e)

