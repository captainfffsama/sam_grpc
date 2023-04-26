#!/bin/sh
kill `cat /run/sam_grpc.pid`
exit 1
