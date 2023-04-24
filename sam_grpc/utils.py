# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-21 16:57:50
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-23 12:39:00
@FilePath: /sam_grpc/lib/utils.py
@Description:
'''
from typing import Union
import torch

import os
import base64

import numpy as np
import cv2

from .proto import dldetection_pb2


def get_img(img_info):
    if os.path.isfile(img_info):
        if not os.path.exists(img_info):
            return None
        else:
            return cv2.imread(img_info)  #ignore
    else:
        img_str = base64.b64decode(img_info)
        img_np = np.fromstring(img_str, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

NP2PROTO_MAP={
    np.dtype("int"): dldetection_pb2.TensorInt,
    np.dtype("bool"): dldetection_pb2.TensorBool,
    np.dtype("float32"): dldetection_pb2.TensorFloat
}

PROTO2NP_MAP={
    dldetection_pb2.TensorInt: np.dtype("int"),
    dldetection_pb2.TensorFloat: np.dtype("float32"),
    dldetection_pb2.TensorBool: np.dtype("bool"),
}

def np2tensor_proto(np_ndarray: Union[np.ndarray,torch.Tensor]):
    if isinstance(np_ndarray,torch.Tensor):
        np_ndarray=np_ndarray.detach().cpu().numpy()
    shape = list(np_ndarray.shape)
    data = np_ndarray.flatten().tolist()
    tensor_pb = NP2PROTO_MAP[np_ndarray.dtype]()
    tensor_pb.shape.extend(shape)
    tensor_pb.data.extend(data)
    return tensor_pb


def tensor_proto2np(tensor_pb):
    np_matrix = np.array(tensor_pb.data,
                         dtype=PROTO2NP_MAP[type(tensor_pb)]).reshape(tensor_pb.shape)
    return np_matrix

def cvImg2ProtoImage(img:Union[str,np.ndarray]) -> dldetection_pb2.Image:
    if isinstance(img,str):
        with open(img,'rb') as f:
            img_b64encode = base64.b64encode(f.read())
        return dldetection_pb2.Image(path=img,imdata=img_b64encode)
    elif isinstance(img,np.ndarray):
        base64_str = cv2.imencode('.jpg', img)[1].tostring()
        img_b64encode = base64.b64encode(base64_str)
        return dldetection_pb2.Image(imdata=img_b64encode)
    else:
        print("error,can not convert image to proto")
        return dldetection_pb2.Image()


def protoImage2cvImg(protoimg:dldetection_pb2.Image) -> np.ndarray:
    if os.path.exists(protoimg.path):
        return cv2.imread(protoimg.path)
    else:
        return get_img(protoimg.imdata)
