# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-21 15:49:00
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-24 12:50:21
@FilePath: /sam_grpc/lib/predictor.py
@Description: a modification for sam predictor
'''
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Optional, Tuple, Union

import numpy as np
import torch

from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

from sam_grpc.proto import dldetection_pb2
from sam_grpc.utils import np2tensor_proto, tensor_proto2np


class InputInferArgs:

    def __init__(self, original_image_size: Tuple[int, int],
                 input_size: Tuple[int, int], features: np.ndarray):
        self.original_size = original_image_size
        self.input_size = input_size
        self.features: np.ndarray = features

    def __str__(self):
        return "original_size:{} \ninput_size:{} \nfeatures size:{}".format(
            self.original_size, self.input_size, self.features.shape)

    def to_proto(self) -> dldetection_pb2.InputInferArgs:
        r = dldetection_pb2.InputInferArgs(features=np2tensor_proto(self.features))
        r.original_size.extend(list(self.original_size))
        r.input_size.extend(list(self.input_size))
        return r

    @classmethod
    def from_proto(cls, proto_cache: dldetection_pb2.InputInferArgs):
        return cls(original_image_size=tuple(proto_cache.original_size),
                   input_size=tuple(proto_cache.input_size),
                   features=tensor_proto2np(proto_cache.features))


class SamPredictorFix:

    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    @torch.no_grad()
    def generate_infer_cache(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> InputInferArgs:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(
            2, 0, 1).contiguous()[None, :, :, :]

        assert (
            len(input_image_torch.shape) == 4
            and input_image_torch.shape[1] == 3
            and max(*input_image_torch.shape[2:])
            == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."

        original_size = image.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])

        input_image = self.model.preprocess(input_image_torch)

        features: torch.Tensor = self.model.image_encoder(input_image)
        return InputInferArgs(original_size, input_size,
                               features.detach().cpu().numpy())

    def predict(
        self,
        infer_args: InputInferArgs,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(
                point_coords, infer_args.original_size)
            coords_torch = torch.as_tensor(point_coords,
                                           dtype=torch.float,
                                           device=self.device)
            labels_torch = torch.as_tensor(point_labels,
                                           dtype=torch.int,
                                           device=self.device)
            coords_torch, labels_torch = coords_torch[
                None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, infer_args.original_size)
            box_torch = torch.as_tensor(box,
                                        dtype=torch.float,
                                        device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input,
                                               dtype=torch.float,
                                               device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            infer_args,
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        torch.cuda.empty_cache()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        infer_args: InputInferArgs,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if isinstance(infer_args.features, np.ndarray):
            features = torch.Tensor(infer_args.features)
        else:
            features = deepcopy(infer_args.features)

        if features.device != self.device:
            features = features.to(self.device)

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks,
                                             infer_args.input_size,
                                             infer_args.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    @property
    def device(self) -> torch.device:
        return self.model.device