#!/bin/bash


# create local disks and then copy
# sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 /dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3 /dev/nvme0n4
# sudo mkfs.ext4 -F /dev/md0
# mkdir -p ~/local
# sudo mount /dev/md0 ~/local
# sudo chmod a+rwx ~/local
# cp -r ~/datasets ~/local/
sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches" && rm -rf /dev/shm/cache

# parameters
num_cpus=`grep -c ^processor /proc/cpuinfo`
num_gpus=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

num_threads=$((num_cpus / num_gpus))  # per GPU
total_mem=`grep MemTotal /proc/meminfo | awk '{print $2}'`  # KB
# cache_mem=31771853
# cache_mem=43725619
# cache_mem=$((194025195 * 10 / 10))
# model=("alexnet" "efficientnet_b0" "mnasnet1_0" "mobilenet_v3_small" "resnet18" "shufflenet_v2_x1_0" "squeezenet1_1" "vgg11")
# batch=(1280 320 512 1280 768 1024 768 192)
# model=("alexnet" "vgg11")
# batch=(1280 192)
# cgroup=(79958511616 76519710720 77093367808 79948283904 78105784320 79218978816 77923913728 75341406208)
deep_batch=56
point_batch=24
yolo_batch=160
# envs=("Liquid" "CoorDL" "DALI")
# envs=("CoorDL" "DALI")
envs=("Liquid")
modes=("load")
# modes=("prep")
dataset_path="/datasets/"

# profile
# conda activate Liquid
# python profile.py --path $dataset_path
# evaluation
for i in {0..10}; do
cache_mem=$((229395813 * $i / 10))
echo $cache_mem
for env in ${envs[@]}; do
conda activate $env
# sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches" && rm -rf /dev/shm/cache

# for mode in ${modes[@]}; do
# echo 57457750016 | sudo tee /sys/fs/cgroup/memory/asplos/memory.limit_in_bytes
# cgexec -g memory:asplos \
# torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus prep_thpt.py \
# --env $env --kind imagenet --mode $mode --path $dataset_path --buffer_size $cache_mem
# done

# for i in "${!model[@]}"; do
# sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches" && rm -rf /dev/shm/cache
# echo "${cgroup[i]}" | sudo tee /sys/fs/cgroup/memory/asplos/memory.limit_in_bytes
# cgexec -g memory:asplos \
# torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus imagenet_end2end_$env.py \
# --path "$dataset_path"imagenet -a "${model[i]}" -j $num_threads -b "${batch[i]}" --buffer_size $cache_mem
# done

# sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches" && rm -rf /dev/shm/cache

# for mode in ${modes[@]}; do
# echo 3000000000 | sudo tee /sys/fs/cgroup/memory/asplos/memory.limit_in_bytes
# cgexec -g memory:asplos \
# torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus prep_thpt.py \
# --env $env --kind cityscapes --mode $mode --path $dataset_path --buffer_size $cache_mem
# done

# sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches" && rm -rf /dev/shm/cache

# cd DeepLabV3Plus-Pytorch_$env
# echo 59567050752 | sudo tee /sys/fs/cgroup/memory/asplos/memory.limit_in_bytes
# cgexec -g memory:asplos \
# torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus main.py \
# --path "$dataset_path"cityscapes --model deeplabv3plus_mobilenet \
# --batch_size $deep_batch --amp --buffer_size $cache_mem --workers $num_threads

# sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches" && rm -rf /dev/shm/cache

# cd ../detectron2_$env/projects/PointRend
# echo 53838139392 | sudo tee /sys/fs/cgroup/memory/asplos/memory.limit_in_bytes
# cgexec -g memory:asplos \
# python train_net.py --config-file configs/SemanticSegmentation/pointrend_semantic_R_50_FPN_1x_cityscapes.yaml \
# --num-gpus $num_gpus DATALOADER.NUM_WORKERS $num_threads DATALOADER.BUFFER_SIZE $cache_mem \
# SOLVER.IMS_PER_BATCH $point_batch DATASETS.PATH "$dataset_path"cityscapes

# sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches" && rm -rf /dev/shm/cache

cd YOLOv5-PyTorch_$env
# echo 62038712320 | sudo tee /sys/fs/cgroup/memory/asplos/memory.limit_in_bytes
# cgexec -g memory:asplos \
python -m torch.distributed.run --nproc_per_node=$num_gpus train.py \
--path "$dataset_path"citypersons --batch-size $yolo_batch --num-threads $num_threads --buffer-size $cache_mem

cd ../
done
done