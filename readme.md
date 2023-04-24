# Introduction
a simple rpc tools for SAM

# Installation
```shell
pip install git+https://github.com/captainfffsama/sam_grpc.git
```

# Getting Started
On Server:
```shell
python -m sam_grpc -h
python -m sam_grpc -c your_sam_grpc_cfg_file_path
```

On client:
```python
import cv2
from sam_grpc import SAMClient

img=cv2.imread("/test_weight/2.jpg")

with SAMClient("127.0.0.1","52018") as client:
    r=client.SAMGetImageEmbedding(img)
    print(r)
    r1=client.SAMGetImageEmbeddingUseCache(img)
    print(r1)
    r2=client.SAMPredict(r)
    print(r2)
    r3=client.SAMPredictUseCache(r1[-1])
    print(r3)
```
