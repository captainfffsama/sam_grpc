# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-21 16:23:16
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-24 13:45:41
@FilePath: /sam_grpc/test.py
@Description:
'''
import cv2

import numpy as np

from segment_anything import sam_model_registry
from sam_grpc.predictor import SamPredictorFix,InputInferArgs
from sam_grpc.utils import np2tensor_proto

import matplotlib.pyplot as plt


def test1():
    sam_checkpoint = "/data/weight/SAM/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda:0"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictorFix(sam)

    img1=cv2.imread("/home/chiebotgpuhq/tmp_space/tmp/1.png")

    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    cache1=predictor.generate_infer_cache(image1)
    print(cache1)

    img2=cv2.imread("/home/chiebotgpuhq/tmp_space/tmp/3.jpg")

    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    cache2=predictor.generate_infer_cache(image2)
    print(cache2)

    mask1,s1,logits1=predictor.predict(cache1,multimask_output=True)

    mask2,s2,logits2=predictor.predict(cache2)

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    b1=logits1[np.argmax(s1), :, :]
    a1=logits1[np.argmax(s1), :, :][None,:,:]
    breakpoint()

    # plt.figure(figsize=(10,10))
    # # plt.imshow(image1)
    # show_mask(a1, plt.gca())
    # plt.axis('off')
    # plt.show()

    # plt.figure(figsize=(10,10))
    # # plt.imshow(image2)
    # show_mask(logits2[np.argmax(s2), :, :], plt.gca())
    # plt.axis('off')
    # plt.show()

    for i, (mask, score) in enumerate(zip(mask1, s1)):
        plt.figure(figsize=(10,10))
        plt.imshow(image1)
        show_mask(mask, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    for i, (mask, score) in enumerate(zip(mask2, s2)):
        plt.figure(figsize=(10,10))
        plt.imshow(image2)
        show_mask(mask, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

import base64

import grpc
from sam_grpc.proto import dldetection_pb2
from sam_grpc.model import InputInferArgs

from sam_grpc.proto import dldetection_pb2_grpc as dld_grpc
from sam_grpc.utils import tensor_proto2np,cvImg2ProtoImage

import cv2


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:52018',options=[('grpc.max_send_message_length',
                  512*1024*1024),
                 ('grpc.max_receive_message_length',
                  512*1024*1024)]) as channel:
        stub = dld_grpc.AiServiceStub(channel)
        img_path = r'/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/00cb74e7b452c399721bff526eb6489c.jpg'
        img_path=r"/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/2.jpg"

        img_proto=cvImg2ProtoImage(img_path)

        req1 =stub.SAMGetImageEmbedding(img_proto)
        print(InputInferArgs.from_proto(req1))

        r2=stub.SAMGetImageEmbeddingUseCache(img_proto)
        print(r2.cache_name+"---"+r2.cache_type)

        r3=stub.SAMPredict()

        # with open("/data/tmp.npy","rb") as fr:
        #     features=np.load(fr)


        # print(features.shape)
        # aaaa=np2tensor_proto(features)
        # print(aaaa.shape)




        # for obj in response.results:
        #     print("type: {} score: {}  box: {}".format(obj.classid, obj.score,
        #                                                obj.rect.x))



if __name__ == '__main__':
    run()