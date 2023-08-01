# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-21 18:15:28
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-08-01 18:00:32
@FilePath: /sam_grpc/sam_grpc/model.py
@Description:
'''
import time
from datetime import datetime

import numpy as np
from cacheout import LFUCache

from .proto import samrpc_pb2
from .proto import samrpc_pb2_grpc as sam_pb2_grpc

from mobile_sam import sam_model_registry
from .predictor import SamPredictorFix
from .utils import tensor_proto2np, np2tensor_proto, protoImage2cvImg, protoTensorIsValid
from .container import InputInferArgs


class SAMGRPCModel(sam_pb2_grpc.SAMServiceServicer):

    def __init__(self,
                 ckpt_path,
                 model_type: str = "vit_t",
                 device: str = "cuda:0") -> None:
        super().__init__()
        self._sam_model = sam_model_registry[model_type](checkpoint=ckpt_path)
        self._sam_model.to(device=device)
        self.predictor = SamPredictorFix(self._sam_model)
        self.cache = LFUCache(maxsize=256,
                              ttl=0,
                              timer=time.time,
                              default=None)

    def _get_cache(self, server_cache_proto: samrpc_pb2.ServerCache):
        cache_type = server_cache_proto.cache_type
        cache_name = server_cache_proto.cache_name
        return self.cache.get(cache_type + cache_name)

    def SAMGetImageEmbedding(self, request,
                             context) -> samrpc_pb2.InputInferArgs:
        image = protoImage2cvImg(request)
        infer_cache = self.predictor.generate_infer_cache(image, "BGR")
        result = infer_cache.to_proto()
        return result

    def SAMGetImageEmbeddingUseCache(
            self, request, context) -> samrpc_pb2.InputInferArgsWithCache:
        image = protoImage2cvImg(request)
        infer_cache = self.predictor.generate_infer_cache(image, "BGR")

        cache_type = "imgEmbedding"
        cache_name = str(
            context.peer()) + datetime.now().strftime("%y%m%d%h%m%s")
        self.cache.set(cache_type + cache_name, infer_cache, ttl=600)

        cache_proto = samrpc_pb2.ServerCache(
            cache_type=cache_type,
            cache_name=cache_name,
        )

        return samrpc_pb2.InputInferArgsWithCache(
            result=infer_cache.to_proto(), cache_idx=cache_proto)

    def SAMPredict(self, request,
                   context) -> samrpc_pb2.SAMPredictResponse:
        infer_args = InputInferArgs.from_proto(request.infer_args)

        point_coords = tensor_proto2np(
            request.point_coords) if protoTensorIsValid(
                request.point_coords) else None
        point_labels = tensor_proto2np(
            request.point_labels) if protoTensorIsValid(
                request.point_labels) else None
        box = tensor_proto2np(request.box) if protoTensorIsValid(
            request.box) else None
        mask_input = tensor_proto2np(request.mask_input) if protoTensorIsValid(
            request.mask_input) else None

        multimask_output = bool(request.multimask_output)
        return_logits = bool(request.return_logits)
        masks, scores, logits = self.predictor.predict(
            infer_args, point_coords, point_labels, box, mask_input,
            multimask_output, return_logits)

        response = samrpc_pb2.SAMPredictResponse(
            masks=np2tensor_proto(masks),
            scores=np2tensor_proto(scores),
            logits=np2tensor_proto(logits),
            status=0)
        return response

    def SAMPredictUseCache(
            self, request: samrpc_pb2.SAMPredictUseCacheRequest,
            context) -> samrpc_pb2.SAMPredictResponseWithCache:
        infer_args_cache = self._get_cache(request.infer_args_cache)
        if infer_args_cache is None:
            r = samrpc_pb2.SAMPredictResponse(status=-1)
            return samrpc_pb2.SAMPredictResponseWithCache(result=r)

        mask_input_cache = self._get_cache(request.mask_input_cache)

        point_coords = tensor_proto2np(
            request.point_coords) if protoTensorIsValid(
                request.point_coords) else None
        point_labels = tensor_proto2np(
            request.point_labels) if protoTensorIsValid(
                request.point_labels) else None
        box = tensor_proto2np(request.box) if protoTensorIsValid(
            request.box) else None

        multimask_output = bool(request.multimask_output)
        return_logits = bool(request.return_logits)
        masks, scores, logits = self.predictor.predict(
            infer_args_cache, point_coords, point_labels, box,
            mask_input_cache, multimask_output, return_logits)

        response = samrpc_pb2.SAMPredictResponse(
            masks=np2tensor_proto(masks),
            scores=np2tensor_proto(scores),
            logits=np2tensor_proto(logits),
            status=0)

        cache_type = "Mask"
        cache_name = str(
            context.peer()) + datetime.now().strftime("%y%m%d%h%m%s")
        self.cache.set(cache_type + cache_name,
                       logits[np.argmax(scores), :, :][None, :, :],
                       ttl=600)

        cache_proto = samrpc_pb2.ServerCache(
            cache_type=cache_type,
            cache_name=cache_name,
        )

        return samrpc_pb2.SAMPredictResponseWithCache(
            result=response, cache_idx=cache_proto)

    def CleanCache(self, request, context):
        if self.cache.has(request.cache_type + request.cache_name):
            self.cache.delete(request.cache_type + request.cache_name)
        return samrpc_pb2.CleanCacheResponse(status=0)
