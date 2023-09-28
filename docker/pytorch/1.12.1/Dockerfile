# pytorch version: 1.12.1
# cuda version: 11.3
# cudnn version: 8
# image type: runtime
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# apt update, install gcc
RUN apt update && apt install build-essential -y && apt-get install libgl1 -y
# apt install package
RUN apt-get install libglib2.0-0 -y

# pip instal sparse package
RUN pip install \
batchgenerators==0.23 \
debugpy==1.6.4 \
dicom2nifti==2.4.7 \
efficientnet-pytorch==0.7.1 \
einops==0.6.1 \
h5py==3.7.0 \
jittor==1.3.6.10 \
matplotlib==3.5.3 \
ml-collections \
MedPy  \
monai==1.1.0 \
opencv-python==4.6.0.66 \
parse==1.19.0 \
pretrainedmodels==0.7.4 \
pydicom==2.3.1 \
scikit-image==0.19.3 \
SimpleITK==2.2.1 \
tensorboard==2.5.0 \
tensorboardX==2.6 \
timm==0.6.12 \
torch-cluster==1.6.0+pt112cu113 \
torch-geometric==2.2.0 \
torch-scatter==2.1.0+pt112cu113 \
torch-sparse==0.6.15+pt112cu113 \
torch-spline-conv==1.2.1+pt112cu113 \
torchio==0.18.90 \
torchprofile==0.0.4 \
trimesh==3.20.0 \
yacs==0.1.8 \
scipy \
timm \
-f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# make folders
RUN mkdir /runtime /dataset
