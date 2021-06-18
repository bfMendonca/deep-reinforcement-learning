#!/bin/bash

#xhost +

XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

docker run -it --rm \
--gpus all \
--privileged \
--env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--env="XAUTHORITY=$XAUTH" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="$XAUTH:$XAUTH" \
-v $(pwd)/../:/home/workspace/deep-reinforcement-learning \
--network host \
pytorch_gym bash
