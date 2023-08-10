from typing import List
import os
from multiprocessing import Process
from subprocess import call
import json
from .cityPersons import CityPersonsConverter
import numpy as np
import cv2 as cv


def extract_imagenet_train_tar(path: str, num_workers: int) -> None:
    '''
    <path>/
        XXX.tar
        YYY.tar
        ...
    '''
    tars = [os.path.join(path, tar) for tar in os.listdir(path)]

    processes = []
    for i in range(num_workers):
        processes.append(Process(target=_worker, args=(tars[i::num_workers],)))

    for process in processes:
        process.start()
    for process in processes:
        process.join()


def _worker(tars: List[str]) -> None:
    for tar in tars:
        dir = os.path.splitext(tar)[0]
        os.makedirs(dir, exist_ok=True)
        call(['tar', '-xf', tar, '-C', dir])
        os.remove(tar)


def cityPersons2COCO(to_path: str, from_path: str, splits: List[str]) -> None:
    '''
    <from_path>/
        gtFine/
        gtCoarse/
        leftImg8bit/
            train/
                aachen/
                bochum/
                ...
            train_extra/
            val/
            test/
        rightImg8bit/
        gtBboxCityPersons/
            train/
                aachen/
                bochum/
                ...
            val/

    <to_path> should be a file path.
    '''

    to_path = os.path.expanduser(to_path)
    from_path = os.path.expanduser(from_path)

    json_dict = {"images": [], "annotations": [], "categories": []}
    id = 0
    image_id = 0

    for split in splits:
        assert split in ['train', 'train_extra', 'val',
                        'train_right', 'train_extra_right', 'val_right'], \
            'No support for {}'.format(split)
        if split in ['train_right', 'val_right']:
            print('split {} not officially supported, please use only for throughput evaluation'.format(split))
        elif split in ['train_extra', 'train_extra_right']:
            print('split {} is a fake dataset, please use only for throughput evaluation'.format(split))

        image_paths = []
        for root, _, files in os.walk(os.path.join(from_path,
                'rightImg8bit' if 'right' in split else 'leftImg8bit', split.rstrip('_right'))):
            for file in files:
                image_paths.append(os.path.join(root, file))
        image_paths.sort()
        num_images = len(image_paths)
        assert num_images == CityPersonsConverter.num_images[split], 'The number of images is not correct, checkout your directory'

        annot_paths = []
        if 'train_extra' in split:
            for root, _, files in os.walk(os.path.join(from_path, 'gtBboxCityPersons', 'train')):
                for file in files:
                    annot_paths.append(os.path.join(root, file))
            annot_paths.sort()
            assert len(annot_paths) == 2975, 'The number of annotations is not correct, checkout your directory'
            annot_paths = (annot_paths * 7)[:19998]
        else:
            for image_path in image_paths:
                head, tail = os.path.split(image_path)
                image_kind = 'leftImg8bit' if 'leftImg8bit' in head else 'rightImg8bit'
                annot_paths.append(os.path.join(head.replace(image_kind, 'gtBboxCityPersons'),
                                                tail.replace(image_kind + '.png', 'gtBboxCityPersons.json')))

        for image_path, annot_path in zip(image_paths, annot_paths):
            file_name = image_path[image_path.find(from_path) + len(from_path) + 1:]
            image = {'file_name': file_name, 'height': 1024, 'width': 2048, 'id': image_id}
            json_dict['images'].append(image)

            with open(annot_path) as f:
                data = json.load(f)

            for instance in data['objects']:
                x, y, w, h = instance['bbox'][:4]
                annot = {'area': w * h, 'iscrowd': 0, 'image_id': image_id, 'bbox': [x, y, w, h],
                        'category_id': CityPersonsConverter.name2id[instance['label']], 'id': id}
                json_dict['annotations'].append(annot)

                id += 1

            image_id += 1

    for k, v in CityPersonsConverter.name2id.items():
        cat = {'supercategory': 'none', 'id': v, 'name': k}
        json_dict['categories'].append(cat)

    json_str = json.dumps(json_dict)
    with open(to_path, 'w') as f:
        f.write(json_str)


def png2others(path: str, num_workers: int, format: str) -> None:
    imgs = []
    for root, _, files in os.walk(path):
        for file in files:
            if '.png' in file:
                imgs.append(os.path.join(root, file))

    processes = []
    for i in range(num_workers):
        processes.append(Process(target=_converter, args=(imgs[i::num_workers], format,)))

    for process in processes:
        process.start()
    for process in processes:
        process.join()


def _converter(imgs: List[str], format: str) -> None:
    for img in imgs:
        with open(img, 'rb') as image_io:
            image = image_io.read()
        image = np.fromstring(image, dtype=np.uint8)
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        _, image = cv.imencode('.' + format, image)
        image = image.tobytes()
        with open(img, 'wb') as image_io:
            image_io.write(image)
