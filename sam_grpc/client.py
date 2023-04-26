# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-24 13:52:17
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-25 18:32:36
@FilePath: /sam_grpc/sam_grpc/client.py
@Description:
'''
from typing import Union, Tuple, Optional

import grpc
import numpy as np

from .proto import samrpc_pb2_grpc as sam_grpc
from .proto import samrpc_pb2
from .container import InputInferArgs, ServerCache
from .utils import cvImg2ProtoImage, np2tensor_proto, tensor_proto2np


class SAMClient(object):

    def __init__(self,
                 host: str,
                 port: str = "52018",
                 max_send_message=512,
                 max_receive_message=512):
        self.host = host
        self.port = port
        self.max_send_message = max_send_message * 1024 * 1024
        self.max_receive_message = max_receive_message * 1024 * 1024

    def __enter__(self):
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.port),
            options=[('grpc.max_send_message_length', self.max_send_message),
                     ('grpc.max_receive_message_length',
                      self.max_receive_message)])
        self.stub = sam_grpc.SAMServiceStub(self.channel)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.channel.close()

    def SAMGetImageEmbedding(self, img: Union[str,
                                              np.ndarray]) -> InputInferArgs:
        img_proto = cvImg2ProtoImage(img)
        response = self.stub.SAMGetImageEmbedding(img_proto)
        return InputInferArgs.from_proto(response)

    def SAMGetImageEmbeddingUseCache(
            self,
            img: Union[str, np.ndarray]) -> Tuple[InputInferArgs, ServerCache]:
        img_proto = cvImg2ProtoImage(img)
        response = self.stub.SAMGetImageEmbeddingUseCache(img_proto)
        return InputInferArgs.from_proto(
            response.result), ServerCache.from_proto(response.cache_idx)

    def SAMPredict(
        self,
        infer_args: InputInferArgs,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray]]:

        proto_args_dict = {
            "infer_args": infer_args.to_proto(),
            "multimask_output": multimask_output,
            "return_logits": return_logits
        }
        if point_coords is not None:
            proto_args_dict["point_coords"] = np2tensor_proto(point_coords)
        if point_labels is not None:
            proto_args_dict["point_labels"] = np2tensor_proto(point_labels)
        if box is not None:
            proto_args_dict["box"] = np2tensor_proto(box)
        if mask_input is not None:
            proto_args_dict["mask_input"] = np2tensor_proto(mask_input)

        request = samrpc_pb2.SAMPredictRequest(**proto_args_dict)
        response = self.stub.SAMPredict(request)
        if response.status != 0:
            return None, None, None
        else:
            return tensor_proto2np(response.masks), tensor_proto2np(
                response.scores), tensor_proto2np(response.logits)

    def SAMPredictUseCache(
        self,
        infer_args: ServerCache,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[ServerCache] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[ServerCache]]:

        proto_args_dict = {
            "infer_args_cache": infer_args.to_proto(),
            "multimask_output": multimask_output,
            "return_logits": return_logits,
        }

        if point_coords is not None:
            proto_args_dict["point_coords"] = np2tensor_proto(point_coords)
        if point_labels is not None:
            proto_args_dict["point_labels"] = np2tensor_proto(point_labels)
        if box is not None:
            proto_args_dict["box"] = np2tensor_proto(box)
        if mask_input is not None:
            proto_args_dict["mask_input_cache"] = mask_input.to_proto()

        request = samrpc_pb2.SAMPredictUseCacheRequest(**proto_args_dict)
        response = self.stub.SAMPredictUseCache(request)
        if response.result.status != 0:
            return None, None, None, None
        else:
            return tensor_proto2np(response.result.masks), tensor_proto2np(
                response.result.scores), tensor_proto2np(
                    response.result.logits), ServerCache.from_proto(
                        response.cache_idx)

    def CleanCache(self, ServerCache):
        response = self.stub.CleanCache(
            samrpc_pb2.CleanCacheRequest(
                ServerCache=ServerCache.to_proto()))
