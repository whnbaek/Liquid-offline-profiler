from typing import List, Tuple, Union
import os
import random
import struct
import json
import webdataset as wds
import numpy as np
import cv2 as cv
from multiprocessing import Process, Queue
from subprocess import call
from .converter import Converter
import nvidia.dali.types as types


class CityPersonsConverter(Converter):
    name2id = {
        'ignore' : 0,
        'pedestrian' : 1,
        'rider' : 2,
        'sitting person' : 3,
        'person (other)' : 4,
        'person group' : 5
    }
    num_images = {'train': 2975, 'train_extra': 19998, 'val': 500,
                'train_right': 2975, 'train_extra_right': 19998, 'val_right': 500}
    info = {
        'ext': ['bbox', 'image', 'image_id', 'label'],
        'dtypes': [types.FLOAT, types.UINT8, types.INT32, types.INT32],
        'is_images': [False, True, False, False],
        'output_type': [None, types.RGB, None, None]
    }

    def _convert(self, pattern: str, split: str, percent: int) -> None:
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
        '''
        assert split in ['train', 'train_extra', 'val',
                        'train_right', 'train_extra_right', 'val_right'], \
            'No support for {}'.format(split)
        if split in ['train_right', 'val_right']:
            print('split {} not officially supported, please use only for throughput evaluation'.format(split))
        elif split in ['train_extra', 'train_extra_right']:
            print('split {} is a fake dataset, please use only for throughput evaluation'.format(split))

        image_paths = []
        for root, _, files in os.walk(os.path.join(self.from_path,
                'rightImg8bit' if 'right' in split else 'leftImg8bit', split.rstrip('_right'))):
            for file in files:
                image_paths.append(os.path.join(root, file))
        image_paths.sort()
        num_images = len(image_paths)
        assert num_images == self.num_images[split], 'The number of images is not correct, checkout your directory'

        random.shuffle(image_paths)

        annot_paths = []
        if 'train_extra' in split:
            for root, _, files in os.walk(os.path.join(self.from_path, 'gtBboxCityPersons', 'train')):
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

        is_firsts = []
        num_raw = 0
        for i in range(num_images):
            if 100 * num_raw < percent * (i + 1):
                num_raw += 1
                is_firsts.append(True)
            else:
                is_firsts.append(False)

        processes = []
        queues = []
        for i in range(self.num_readers):
            queues.append(Queue(2))
            processes.append(Process(target=self._reader, args=(queues[-1], self.formats,
                                    image_paths[i::self.num_readers], annot_paths[i::self.num_readers], is_firsts[i::self.num_readers])))
            processes[-1].start()

        image_shapes = []
        with wds.ShardWriter(pattern, encoder = False) as writer:
            for i in range(num_images):
                image, image_shape, bbox, label = queues[i % self.num_readers].get()
                image_shapes.append(image_shape)
                sample = {
                    '__key__': "sample%d" % i,
                    'bbox' : bbox,
                    'image': image,
                    'image_id': struct.pack('i', i),
                    'label': label
                }
                writer.write(sample)

        for process in processes:
            process.join()

        i, acc = 0, 0
        while os.path.exists(pattern % i):
            tar_path = pattern % i
            idx_path = os.path.splitext(tar_path)[0] + '.idx'

            call(['wds2idx', tar_path, idx_path])

            with open(idx_path, 'r') as f:
                lines = f.readlines()

            for num in range(acc, acc + len(lines) - 1):
                param = lines[num - acc + 1].split(' ')
                param[-1] = param[-1].rstrip('\n')
                param.insert(0, str(int(is_firsts[num])))
                param.insert(5, '0')
                param.insert(6, '0')
                param.insert(7, '0')
                param.insert(12, str(image_shapes[num][0]))
                param.insert(13, str(image_shapes[num][1]))
                param.insert(14, str(image_shapes[num][2]))
                param.insert(19, '0')
                param.insert(20, '0')
                param.insert(21, '0')
                param.append('0')
                param.append('0')
                param.append('0\n')
                lines[num - acc + 1] = ' '.join(param)

            with open(idx_path, 'w') as f:
                f.writelines(lines)

            i += 1
            acc += len(lines) - 1


    def _convert_part(self, pattern: str, percent: int) -> None:
        self._convert(pattern, 'val', percent)


    @staticmethod
    def _reader(queue: Queue, formats: Tuple[Union[str, None], Union[str, None]],
                image_paths: List[str], annot_paths: List[str], is_firsts: List[bool]) -> None:
        for image_path, annot_path, is_first in zip(image_paths, annot_paths, is_firsts):
            format = formats[0] if is_first else formats[1]

            with open(image_path, 'rb') as image_io:
                image = image_io.read()

            image_shape = (0, 0, 0)

            if format is not None:
                image = np.fromstring(image, dtype = np.uint8)
                image = cv.imdecode(image, cv.IMREAD_COLOR)

                if format == 'raw':
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    image_shape = image.shape
                else:
                    _, image = cv.imencode('.' + format, image)
                    image_shape = (0, 0, 0)

                image = image.tobytes()

            with open(annot_path) as f:
                data = json.load(f)
            bbox = []
            label = []
            for instance in data['objects']:
                instance['bbox'][2] += instance['bbox'][0]
                instance['bbox'][3] += instance['bbox'][1]
                bbox.extend([float(i) for i in instance['bbox']])
                label.append(CityPersonsConverter.name2id[instance['label']])

            bbox = struct.pack('%sf' % len(bbox), *bbox)
            label = struct.pack('%si' % len(label), *label)

            queue.put((image, image_shape, bbox, label))
