# Introduction
a simple rpc tools for [SAM](https://github.com/facebookresearch/segment-anything)

# Installation
install on server need `torch`,[SAM](https://github.com/facebookresearch/segment-anything)
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
from sam_grpc import InputInferArgs,ServerCache

img=cv2.imread("/test_weight/2.jpg")

with SAMClient("127.0.0.1","52018") as client:
    img=cv2.imread(img_path)
    r=client.SAMGetImageEmbedding(img)
    print("result is:",r)
    r1=client.SAMGetImageEmbeddingUseCache(img)
    print("result is:",r1[0],r1[1])

    r2=client.SAMPredict(r)
    print("result shape:{} \n {} \n {}".format(r2[0].shape,r2[1].shape,r2[2].shape))

    r3=client.SAMPredictUseCache(r1[1])
    print("result shape:{} \n {} \n {}".format(r3[0].shape,r3[1].shape,r3[2].shape))
    print("mask cache name:",r3[-1])
```
# Thanks
Thanks to Meta for open-sourcing their [excellent work](https://github.com/facebookresearch/segment-anything).
