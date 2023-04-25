# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-24 14:16:12
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-24 16:24:29
@FilePath: /sam_grpc/sam_grpc/container.py
@Description:
'''

from typing import Optional, Tuple, Union

import numpy as np

from .proto import samrpc_pb2
from .utils import np2tensor_proto, tensor_proto2np

class InputInferArgs:

    def __init__(self, original_image_size: Tuple[int, int],
                 input_size: Tuple[int, int], features: np.ndarray):
        self.original_size = original_image_size
        self.input_size = input_size
        self.features: np.ndarray = features

    def __repr__(self):
        return "InputInferArgs:\noriginal_size:{} \ninput_size:{} \nfeatures size:{}".format(
            self.original_size, self.input_size, self.features.shape)

    def to_proto(self) -> samrpc_pb2.InputInferArgs:
        r = samrpc_pb2.InputInferArgs(features=np2tensor_proto(self.features))
        r.original_size.extend(list(self.original_size))
        r.input_size.extend(list(self.input_size))
        return r

    @classmethod
    def from_proto(cls, proto: samrpc_pb2.InputInferArgs):
        return cls(original_image_size=tuple(proto.original_size),
                   input_size=tuple(proto.input_size),
                   features=tensor_proto2np(proto.features))


class ServerCache:
    def __init__(self,cache_name,cache_type):
        self.cache_name=cache_name
        self.cache_type=cache_type

    def __repr__(self):
        return "ServerCache:\ncache_name:{} \ncache_type:{}".format(self.cache_name,self.cache_type)

    def to_proto(self) -> samrpc_pb2.ServerCache:
        r = samrpc_pb2.ServerCache(cache_name=self.cache_name,
                                        cache_type=self.cache_type)
        return r

    @classmethod
    def from_proto(cls,proto:samrpc_pb2.ServerCache):
        return cls(cache_name=proto.cache_name,
                   cache_type=proto.cache_type)