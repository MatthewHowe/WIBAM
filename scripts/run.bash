#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/app/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    --gpus all \
    --ipc host \
    matthewhowe/wibam \
    ${@:-bash}