# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-04-24 15:04:42
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-04-24 15:12:12
@FilePath: /sam_grpc/sam_grpc/__main__.py
@Description:
'''

import sys
import os
import argparse
from concurrent import futures
from pprint import pprint
from datetime import datetime
import asyncio
import pid
from pid.decorator import pidfile
import pkgutil
import yaml

import grpc
from .proto import samrpc_pb2_grpc as sam_grpc
from .model import SAMGRPCModel
from . import base_config as config_manager


async def run_server(args):
    if os.path.exists(args):
        config_manager.merge_param(args)
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
    model = SAMGRPCModel(**detector_params)
    sam_grpc.add_SAMServiceServicer_to_server(model, server)

    server.add_insecure_port("{}:{}".format(grpc_args['host'],
                                            grpc_args['port']))
    await server.start()
    print('sam gprc server init done')
    await server.wait_for_termination()

@pidfile(pidname='sam_grpc')
def start_server(args):
    asyncio.run(run_server(args))

def show_cfg_exam():
    exam_byte = pkgutil.get_data(__package__, 'sam_cfg.yaml')
    from pprint import pprint
    pprint(yaml.load(exam_byte, Loader=yaml.FullLoader))

def main():
    parser = argparse.ArgumentParser(description="grpc调用sam,需要配置文件")
    parser.add_argument("-c", "--config", type=str, default="", help="配置文件地址")
    parser.add_argument("-s",
                        "--show_cfg",
                        action="store_true",
                        help="展示配置文件示例")
    options = parser.parse_args()
    if options.config:
        start_server(options.config)
    if options.show_cfg:
        show_cfg_exam()

if __name__ == "__main__":
    rc = 1
    try:
        main()
    except Exception as e:
        print('Error: %s' % e, file=sys.stderr)
    sys.exit(rc)