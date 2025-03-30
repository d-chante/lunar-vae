#!/bin/bash

source common.sh

docker run -it --entrypoint /bin/bash \
	-v $LUNAR_VAE:/workspace/lunar-vae \
	-v $OUTPUTS:/workspace/outputs \
	-v $DATA:/workspace/datasets \
	$CONTAINER_NAME