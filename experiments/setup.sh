#!/bin/bash

# set your dataset path
dataset_path="~/datasets"

# install apt packages
sudo apt update
sudo apt install mdadm --no-install-recommends

# install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh -b
source $HOME/anaconda3/bin/activate
conda init
source ~/.bashrc

# setup Liquid
conda create -y -n Liquid python=3.7
conda activate Liquid
conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install wheelhouse/nvidia_dali_cuda110-1.15.0-12345_Liquid-py3-none-manylinux2014_x86_64.whl --quiet
pip install cityscapesscripts visdom opencv-python webdataset psutil --quiet
pip install -e detectron2_Liquid --quiet

# setup CoorDL
conda create -y -n CoorDL python=3.7
conda activate CoorDL
conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install wheelhouse/nvidia_dali_cuda110-1.15.0-12345_CoorDL-py3-none-manylinux2014_x86_64.whl --quiet
pip install cityscapesscripts visdom opencv-python webdataset psutil --quiet
pip install -e detectron2_CoorDL --quiet

# setup DALI
conda create -y -n DALI python=3.7
conda activate DALI
conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install wheelhouse/nvidia_dali_cuda110-1.15.0-12345_DALI-py3-none-manylinux2014_x86_64.whl --quiet
pip install cityscapesscripts visdom opencv-python webdataset psutil --quiet
pip install -e detectron2_DALI --quiet

# download
mkdir -p $dataset_path
cd $dataset_path

wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=whnbaek&password=dngus09%2624&submit=Login' https://www.cityscapes-dataset.com/login/
cityscapes=()
wget --load-cookies cookies.txt --content-disposition -q https://www.cityscapes-dataset.com/file-handling/?packageID=1 &
cityscapes[0]=$!
wget --load-cookies cookies.txt --content-disposition -q https://www.cityscapes-dataset.com/file-handling/?packageID=2 &
cityscapes[1]=$!
wget --load-cookies cookies.txt --content-disposition -q https://www.cityscapes-dataset.com/file-handling/?packageID=3 &
cityscapes[2]=$!
wget --load-cookies cookies.txt --content-disposition -q https://www.cityscapes-dataset.com/file-handling/?packageID=4 &
cityscapes[3]=$!
wget --load-cookies cookies.txt --content-disposition -q https://www.cityscapes-dataset.com/file-handling/?packageID=5 &
cityscapes[4]=$!
wget --load-cookies cookies.txt --content-disposition -q https://www.cityscapes-dataset.com/file-handling/?packageID=7 &
cityscapes[5]=$!
wget --load-cookies cookies.txt --content-disposition -q https://www.cityscapes-dataset.com/file-handling/?packageID=28 &
cityscapes[6]=$!
imagenet=()
wget -q https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar &
imagenet[0]=$!
wget -q https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar &
imagenet[1]=$!

for pid in ${cityscapes[*]}; do
    wait $pid
done

# unzip cityscapes
mkdir -p cityscapes
unzip -nq gtFine_trainvaltest.zip -d cityscapes
unzip -nq gtCoarse.zip -d cityscapes
unzip -nq leftImg8bit_trainvaltest.zip -d cityscapes
unzip -nq leftImg8bit_trainextra.zip -d cityscapes
unzip -nq rightImg8bit_trainvaltest.zip -d cityscapes
unzip -nq rightImg8bit_trainextra.zip -d cityscapes
unzip -nq gtBbox_cityPersons_trainval.zip -d cityscapes

# process cityscapes dataset
CITYSCAPES_DATASET="cityscapes" csCreateTrainIdLabelImgs

for pid in ${imagenet[*]}; do
    wait $pid
done

# unzip ImageNet
mkdir -p imagenet/train
tar -xf ILSVRC2012_img_train.tar -C imagenet/train
mkdir -p imagenet/val
tar -xf ILSVRC2012_img_val.tar -C imagenet/val
cd imagenet/val
wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
source valprep.sh
rm valprep.sh
cd ../../
rm *.tar *.zip cookies.txt index.html

# prepare for datasets
python ~/asplos23/prepare_datasets.py
