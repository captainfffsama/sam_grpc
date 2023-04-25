python -m grpc_tools.protoc -I ./ --proto_path=./samrpc.proto --python_out=.  --grpc_python_out=./ samrpc.proto
