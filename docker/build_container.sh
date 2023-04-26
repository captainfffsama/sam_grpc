###
 # @Author: captainfffsama
 # @Date: 2023-04-12 12:24:47
 # @LastEditors: captainfffsama tuanzhangsama@outlook.com
 # @LastEditTime: 2023-04-26 17:24:58
 # @FilePath: /sam_grpc/docker/build_container.sh
 # @Description:
###
docker build -t sam_rpc:v1 . --no-cache
docker run -itd --runtime=nvidia --gpus all -p 2814:2814 -p 52018:52018 --name  sam_server sam_rpc:v1 --restart=always
docker update sam_server --restart=always