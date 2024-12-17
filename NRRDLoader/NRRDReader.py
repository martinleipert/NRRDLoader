from nrrd.reader import read
from typing import List
import numpy as np
from .Segment import Segment
from .PadToMinSize import pad_to_min_size

"""
Martin Leipert
martin.leipert@th-deg.de

29.08.2022

Reads an nrrd file and creates a mask 
Labels have to be defined previously 
"""


class NRRDReader:
    def __init__(self, labels: List, auto_background=False, bg_index=None, min_size=None, swap=True):
        """
        To load multiple data with a fixed setup
        :param labels: Labels to load => Indices in the list are then set as class labels
        :param auto_background: Calculate an automatic background (last index in list)
        """
        if not NRRDReader.validate_labels(labels):
            raise Exception("Label validation failed")
        self.__labels = labels
        self.__auto_background = auto_background
        self.__min_size = min_size

        # Assign an index to each label
        self.__label_indices = dict(zip(labels, range(len(labels))))

        if auto_background:
            self.__bg_index = len(labels)
        else:
            self.__bg_index = bg_index

        self.__swap = swap

    def read_nrrd(self, path_to_file: str):
        data, header = read(path_to_file)
        data = data.astype(np.uint8)

        if self.__swap == True:
            if len(data.shape) == 3:
                data = np.swapaxes(data, 0, 2)
            else:
                data = np.swapaxes(data, 1, 3)

        segments = Segment.parse_segments(header)
        segment_keys = list(map(lambda x: x.get_name(), segments))
        segment_dict = dict(zip(segment_keys, segments))

        # Calculate array size
        num_data = len(self.__labels)
        if self.__auto_background:
            num_data += 1

        # Allocate mask
        if len(data.shape) == 4:
            mask_size = (num_data, data.shape[1], data.shape[2], data.shape[3])
        else:
            mask_size = (num_data, data.shape[0], data.shape[1], data.shape[2])
        mask = np.zeros(mask_size, np.uint8)

        for label in self.__labels:
            label_index = self.__label_indices[label]
            mask[label_index] = np.where(NRRDReader.extract_segment(data, label, segment_dict), np.uint8(1), np.uint8(0))

            if self.__auto_background:
                mask[self.__bg_index] = np.subtract(mask[self.__bg_index], extraced_mask)

        if self.__min_size is not None:
            mask = pad_to_min_size(mask, self.__min_size, dims=(1, 2, 3), pad_constant=False)

        if self.__auto_background:
            tmp = np.invert(np.any(mask[:self.__bg_index], axis=0))
            mask[self.__bg_index] = tmp

        mask = mask.astype(np.float16)

        return mask

    def read_nrrd_torchio(self, path_to_file: str):

        mask = self.read_nrrd(path_to_file)

        return mask, np.eye(4)

    def get_background_index(self):
        return self.__bg_index

    @classmethod
    def extract_segment(cls, data, label, segment_dict):
        if isinstance(label, list) or isinstance(label, tuple):
            tmp_array = np.full(data.shape[-3:], False, dtype=np.bool)
            for single_label in label:
                np.logical_or(tmp_array, NRRDReader.extract_segment(data, single_label, segment_dict),
                              out=tmp_array)
                # tmp_array = np.logical_or(tmp_array, NRRDReader.extract_segment(data, single_label, segment_dict))
        else:
            label_segment = segment_dict[label]
            label_layer = label_segment.get_layer()
            label_label = label_segment.get_label_value()
            if len(data.shape) == 3:
                tmp_array = (data == label_label)
            else:
                tmp_array = (data[label_layer] == label_label)
        return tmp_array

    @classmethod
    def validate_labels(cls, labels):
        for label in labels:
            if isinstance(label, tuple) or isinstance(label, list):
                # Recursively validate the elements
                NRRDReader.validate_labels(label)
            elif not isinstance(label, str):
                raise TypeError(f"label {label} must by of type string or tuple of strings"
                                f" but is of type {label.__type__()}")
        return True
