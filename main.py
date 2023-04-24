# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-21 15:50:02
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-23 12:11:41
@FilePath: /sam_grpc/main.py
@Description:
'''
import os
import argparse
from concurrent import futures
from pprint import pprint
from datetime import datetime
import asyncio
import pid
from pid.decorator import pidfile

import grpc
from sam_grpc.proto import dldetection_pb2_grpc as dld_grpc
from sam_grpc.model import SAMGRPCModel
import sam_grpc.base_config as config_manager


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-mt", "--model_type", type=str, default="")
    parser.add_argument("-mw", "--model_weight", type=str, default="")
    parser.add_argument("-c", "--cfg", type=str, default="")
    args = parser.parse_args()
    return args


async def main(args):
    if os.path.exists(args.cfg):
        config_manager.merge_param(args.cfg)
    args_dict: dict = config_manager.param
    print("current time is: ", datetime.now())
    print("pid file save in {}".format(pid.DEFAULT_PID_DIR))

    pprint(args_dict)

    grpc_args = args_dict['grpc_args']
    detector_params = args_dict['model_params']
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=grpc_args['max_workers']),
        options=[('grpc.max_send_message_length',
                  grpc_args['max_send_message_length']),
                 ('grpc.max_receive_message_length',
                  grpc_args['max_receive_message_length'])])
    if os.path.exists(args.model_type) and os.path.exists(args.model_weight):
        detector_params['model_type'] = args.model_type
        detector_params['ckpt_path'] = args.model_weight
    model = SAMGRPCModel(**detector_params)
    dld_grpc.add_AiServiceServicer_to_server(model, server)

    server.add_insecure_port("{}:{}".format(grpc_args['host'],
                                            grpc_args['port']))
    await server.start()
    print('sam gprc server init done')
    await server.wait_for_termination()

@pidfile(pidname='sam_grpc')
def run():
    args = parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    run()
