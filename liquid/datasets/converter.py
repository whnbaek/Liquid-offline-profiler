from __future__ import annotations
from typing import Tuple, Union
import random
import os
import shutil


class Converter:
    info = {
        'ext': [],
        'dtypes': [],
        'is_images': [],
        'output_type': []
    }

    def __init__(self, to_path: str, from_path: str, seed: int = 12345, num_readers: int = 1,
                 formats: Tuple[Union[str, None], Union[str, None]] = ('raw', None)) -> None:
        '''
        <to_path>/
            "<split>_<percent>"/
            ...
        '''
        self.to_path = os.path.expanduser(to_path)
        self.from_path = os.path.expanduser(from_path)
        self.num_readers = num_readers
        self.formats = formats
        random.seed(seed)


    def convert(self, split: str, percent: int) -> Converter:
        self.to_dir = os.path.join(self.to_path, '{}_{}'.format(split, percent))
        if os.path.isdir(self.to_dir):
            return self
        os.makedirs(self.to_dir)

        pattern = os.path.join(self.to_dir, '%d.tar')
        if split == 'part':
            self._convert_part(pattern, percent)
        else:
            self._convert(pattern, split, percent)

        return self


    def _convert(self, pattern: str, split: str, percent: int) -> None:
        raise NotImplementedError('Please implement convert part of the dataset')


    def _convert_part(self, pattern: str, percent: int) -> None:
        raise NotImplementedError('Please implement convert part of the dataset')


    def delete(self, split: str, percent: int) -> Converter:
        to_dir = os.path.join(self.to_path, '{}_{}'.format(split, percent))
        shutil.rmtree(to_dir, ignore_errors=True)

        return self
