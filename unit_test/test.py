# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-21 16:23:16
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-07-17 16:23:35
@FilePath: /sam_grpc/unit_test/test.py
@Description:
'''

import cv2
from sam_grpc.client import SAMClient
from sam_grpc import InputInferArgs,ServerCache,cv2imread

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    img_path = r'/home/chiebotgpuhq/MyCode/python/pytorch/mmdet_grpc/test_weight/00cb74e7b452c399721bff526eb6489c.jpg'
    client = SAMClient("127.0.0.1","52018")
    with client:
        img=cv2imread(img_path,cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        r=client.SAMGetImageEmbedding(img)
        print("result is:",r)
        r1=client.SAMGetImageEmbeddingUseCache(img)
        print("result is:",r1[0],r1[1])

        r2=client.SAMPredict(r)
        print("result shape:{} \n {} \n {}".format(r2[0].shape,r2[1].shape,r2[2].shape))

        r3=client.SAMPredictUseCache(r1[1])
        print("result shape:{} \n {} \n {}".format(r3[0].shape,r3[1].shape,r3[2].shape))
        print("mask cache name:",r3[-1])



if __name__ == '__main__':
    run()