# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-21 16:57:50
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-24 17:08:21
@FilePath: /sam_grpc/sam_grpc/utils.py
@Description:
'''
from typing import Union

import os
import base64

import numpy as np
import cv2

from .proto import samrpc_pb2

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


NP2PROTO_MAP = {
    np.dtype("int"): samrpc_pb2.TensorInt,
    np.dtype("bool"): samrpc_pb2.TensorBool,
    np.dtype("float32"): samrpc_pb2.TensorFloat
}

PROTO2NP_MAP = {
    samrpc_pb2.TensorInt: np.dtype("int"),
    samrpc_pb2.TensorFloat: np.dtype("float32"),
    samrpc_pb2.TensorBool: np.dtype("bool"),
}


def np2tensor_proto(
    np_ndarray: Union[np.ndarray, "torch.Tensor"]
) -> Union[samrpc_pb2.TensorInt, samrpc_pb2.TensorFloat,
           samrpc_pb2.TensorBool]:
    """
    Converts a NumPy ndarray or a PyTorch tensor to a tensor protobuf message.

    Args:
        np_ndarray (Union[np.ndarray, torch.Tensor]): The NumPy ndarray or PyTorch tensor to convert.

    Returns:
        Union[samrpc_pb2.TensorInt, samrpc_pb2.TensorFloat, samrpc_pb2.TensorBool]:
        A tensor protobuf message containing the data from the input ndarray or tensor.
    """
    if not isinstance(np_ndarray, np.ndarray) and hasattr(np_ndarray, "detach"):
        np_ndarray = np_ndarray.detach().cpu().numpy()
    shape = list(np_ndarray.shape)
    data = np_ndarray.flatten().tolist()
    tensor_pb = NP2PROTO_MAP[np_ndarray.dtype]()
    tensor_pb.shape.extend(shape)
    tensor_pb.data.extend(data)
    return tensor_pb


def tensor_proto2np(
    tensor_pb: Union[samrpc_pb2.TensorInt, samrpc_pb2.TensorFloat,
                     samrpc_pb2.TensorBool]
) -> np.ndarray:
    """Converts a Protocol Buffer tensor to a NumPy array.

    Args:
        tensor_pb: A Protocol Buffer tensor of type `samrpc_pb2.TensorInt`,
            `samrpc_pb2.TensorFloat`, or `samrpc_pb2.TensorBool`.

    Returns:
        A NumPy array with the same data and shape as the input tensor.
    """
    np_matrix = np.array(tensor_pb.data,
                         dtype=PROTO2NP_MAP[type(tensor_pb)]).reshape(
                             tensor_pb.shape)
    return np_matrix


def cvImg2ProtoImage(img: Union[str, np.ndarray]) -> samrpc_pb2.Image:
    """
    Converts an image to a protocol buffer message.

    Args:
        img (Union[str, np.ndarray]): The image to be converted. It can be either
            the path to an image file or a NumPy array.

    Returns:
        samrpc_pb2.Image: The protocol buffer message representing the
        converted image.

    Raises:
        None.

    Example:
        >>> img_path = 'path/to/image.jpg'
        >>> proto_img = cvImg2ProtoImage(img_path)
        >>> assert isinstance(proto_img, samrpc_pb2.Image)
    """
    # Function body goes here

    if isinstance(img, str):
        with open(img, 'rb') as f:
            img_b64encode = base64.b64encode(f.read())
        return samrpc_pb2.Image(path=img, imdata=img_b64encode)
    elif isinstance(img, np.ndarray):
        base64_str = cv2.imencode('.jpg', img)[1].tostring()
        img_b64encode = base64.b64encode(base64_str)
        return samrpc_pb2.Image(imdata=img_b64encode)
    else:
        print("error,can not convert image to proto")
        return samrpc_pb2.Image()


def protoImage2cvImg(protoimg: samrpc_pb2.Image) -> np.ndarray:
    """
    Convert a protobuf image to a numpy array compatible with OpenCV.

    Args:
        protoimg: A protobuf image object containing either a path or raw data.

    Returns:
        A 3-dimensional numpy array representing the image, with shape (height, width, channels).

    Raises:
        FileNotFoundError: If the path specified in protoimg does not exist.
    """
    if os.path.exists(protoimg.path):
        return cv2.imread(protoimg.path)
    else:
        return get_img(protoimg.imdata)


def protoTensorIsValid(
    proto_tensor: Union[samrpc_pb2.TensorInt, samrpc_pb2.TensorFloat,
                        samrpc_pb2.TensorBool]
) -> bool:
    """
    Check if a protobuf tensor is valid by verifying that its data and shape fields are not empty.

    Args:
        proto_tensor: A protobuf tensor, which can be of type TensorInt, TensorFloat, or TensorBool.

    Returns:
        A boolean value indicating whether the tensor is valid or not.
    """
    return proto_tensor.data and proto_tensor.shape
