#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
# import torch

# import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog  # , DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.point_rend import add_pointrend_config  # , ColorAugSSDTransform,

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from glob import glob
import psutil

# def build_sem_seg_train_aug(cfg):
#     augs = [
#         T.ResizeShortestEdge(
#             cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
#         )
#     ]
#     if cfg.INPUT.CROP.ENABLED:
#         augs.append(
#             T.RandomCrop_CategoryAreaConstraint(
#                 cfg.INPUT.CROP.TYPE,
#                 cfg.INPUT.CROP.SIZE,
#                 cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
#                 cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
#             )
#         )
#     if cfg.INPUT.COLOR_AUG_SSD:
#         augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
#     augs.append(T.RandomFlip())
#     return augs


@pipeline_def
def create_dali_pipeline(paths, index_paths, initial_fill, num_shards, shard_id):
    images, labels = fn.readers.webdataset(
            ext=['image', 'seg_label'], paths=paths, index_paths=index_paths,
            initial_fill=initial_fill, stick_to_shard=True, missing_component_behavior='error',
            pad_last_batch=True, random_shuffle=True, shard_id=shard_id, num_shards=num_shards,
            tensor_init_bytes=0, name='Reader')
    images = fn.decoders.image(images, device_memory_padding=0, host_memory_padding=0,
                               hybrid_huffman_threshold=0, device='mixed')
    labels = fn.decoders.image(labels, device_memory_padding=0, host_memory_padding=0,
                               hybrid_huffman_threshold=0, output_type=types.GRAY,
                               device='mixed')
    pos_xs = fn.random.uniform(range=[0.0, 1.0])
    pos_ys = fn.random.uniform(range=[0.0, 1.0])
    mirrors = fn.random.coin_flip()
    shorters = fn.random.uniform(values=[512, 768, 1024, 1280, 1536, 1792, 2048])
    images = fn.resize(images, max_size=4096, resize_shorter=shorters)
    labels = fn.resize(labels, interp_type=types.INTERP_NN, max_size=4096, resize_shorter=shorters)
    images = fn.crop_mirror_normalize(images, crop=[512, 1024], crop_pos_x=pos_xs, crop_pos_y=pos_ys,
                                      dtype=types.UINT8, mirror=mirrors)
    labels = fn.crop_mirror_normalize(labels, crop=[512, 1024], crop_pos_x=pos_xs, crop_pos_y=pos_ys,
                                      dtype=types.UINT8, mirror=mirrors)
    return images, labels


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
        #     mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        # else:
        #     mapper = None
        # return build_detection_train_loader(cfg, mapper=mapper)
        rank = comm.get_rank()
        world_size = comm.get_world_size()
        path = os.path.expanduser(cfg.DATASETS.PATH)
        dir_paths = glob(os.path.join(path, 'train_*'))
        paths = []
        for dir_path in dir_paths:
            for tar in os.listdir(dir_path):
                if '.tar' in tar:
                    paths.append(os.path.join(dir_path, tar))
        paths.sort()
        index_paths = [path.replace('.tar', '.idx') for path in paths]

        nsamples = 0
        for index_path in index_paths:
            with open(index_path, 'r') as f:
                nsamples += int(f.readline().rstrip('\n').split(' ')[1])
        dataset_size = 0
        for path in paths:
            dataset_size += os.path.getsize(path)

        buffer_ratio = cfg.DATALOADER.BUFFER_SIZE / (dataset_size // 1024)
        initial_fill = int(nsamples * buffer_ratio / world_size)

        pipe = create_dali_pipeline(
                batch_size=cfg.SOLVER.IMS_PER_BATCH, num_threads=cfg.DATALOADER.NUM_WORKERS,
                device_id=rank, paths=paths, index_paths=index_paths, initial_fill=initial_fill,
                num_shards=world_size, shard_id=rank)
        pipe.build()
        train_loader = DALIGenericIterator(pipe, output_map=['image', 'label'], reader_name='Reader',
                                           auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        return train_loader


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    if comm.is_main_process():
        start_bytes = psutil.virtual_memory()._asdict()['used']
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    ret = trainer.train()
    if comm.is_main_process():
        bytes_used = psutil.virtual_memory()._asdict()['used'] - start_bytes
        print(bytes_used)
    return ret


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print('env: Liquid, model: PointRend')
    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
