#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/WIBAM/:rw"

# If dataset is not within the WIBAM directory link it to docker here:
# Give path from base ie "/home/user/Downloads/WIBAM"
# data_volumes="--volume=ABSOLUTE_PATH_TO_DATASET:/WIBAM/data/wibam:rw"

docker run --rm -ti \
    $cmp_volumes \
    # $data_volumes \
    -it \
    --gpus all \
    --ipc host \
    matthewhowe/wibam \
    ${@:-bash}