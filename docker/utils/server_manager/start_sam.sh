#!/bin/sh
LOG_BASE_DIR="/root/mount/logs"
LOG_DIR="$LOG_BASE_DIR/sam_grpc"
LOG_PATH="$LOG_DIR/$(date "+%Y%m%d").log"
if [ ! -d $LOG_DIR ]
then
    echo $LOG_DIR" is not exist,will be created"
    mkdir -p $LOG_DIR
fi
if [ ! -f $LOG_PATH ]
then
    touch $LOG_PATH
fi
if [ -f "/run/sam_grpc.pid" ]
then
    kill `cat /run/sam_grpc.pid`
    rm /run/sam_grpc.pid
fi
echo "log save path is: "$LOG_PATH
nohup /opt/conda/bin/python -u -m sam_grpc -c /root/cfg.yaml >> $LOG_PATH  2>&1 &
exit 1
