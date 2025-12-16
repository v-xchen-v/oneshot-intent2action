#!/bin/bash
# current folder as WORD_DIR
CURRENT_DIR=$(pwd)

set -e

sudo docker run -itd --name foundation_pose \
    --gpus all \
    --network=host \
    -v $CURRENT_DIR:/root/workspace/main \
    -w /root/workspace/main \
    shingarey/foundationpose_custom_cuda121:latest \
    /bin/bash