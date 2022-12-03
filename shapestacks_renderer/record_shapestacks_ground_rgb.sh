#!/bin/sh
echo "Recording ShapeStacks scenarios from MuJoCo."

# filter parameters
# $1: dataset name
# $2: height filter

# paths
DATASET_NAME=$1
DATASET_ROOT_DIR="${SHAPESTACKS_CODE_HOME}/data/${DATASET_NAME}"
MJCF_ROOT_DIR="${DATASET_ROOT_DIR}/mjcf"
RECORD_ROOT_DIR="${DATASET_ROOT_DIR}/recordings"

# recording options
TIME=3
FPS=8
MAX_FRAMES=3
RES="128 128"
CAMERAS="cam_1 cam_2 cam_3"
FORMAT="rgb depth"
MAX_LIGHTS=1
MAX_WALLTEX=1
MAX_FLOORTEX=1
COLOR_MODE="original"

# helper functions
create_params()
{
  model_path=$1   # $1: model_path
  record_path=$2  # $2: record_path

  # randomize light and textures
  lightid=`python -c "import random; print(random.randint(0,${MAX_LIGHTS}-1))"`
  walltex=`python -c "import random; print(random.randint(0,${MAX_WALLTEX}-1))"`
  floortex=`python -c "import random; print(random.randint(0,${MAX_FLOORTEX}-1))"`
  # echo $lightid $walltex $floortex

  echo "--mjmodel_path ${model_path} --record_path ${record_path} \
      --mjsim_time ${TIME} --fps ${FPS} --max_frames ${MAX_FRAMES} \
      --res ${RES} --cameras ${CAMERAS} \
      --formats ${FORMAT} \
      --color_mode ${COLOR_MODE}"
}

###
# Main body
###

# directory setup
# mkdir ${DATASET_ROOT_DIR}
# mkdir ${RECORD_ROOT_DIR}

# main loop over all simulation environments to record
# for env_file in `ls ${MJCF_ROOT_DIR} | grep env_ | grep ${FILTER}`; do
for env_file in `ls ${MJCF_ROOT_DIR} | grep env_`; do
  echo "Recording ${env_file%".xml"} ..."

  # set up directory
  record_dir=${RECORD_ROOT_DIR}/${env_file%".xml"}
  log_file=${record_dir}/log.txt
  mkdir ${record_dir}

  # create params and render
  params=$(create_params "${MJCF_ROOT_DIR}/$env_file" $record_dir)
  # echo $params
  # LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOpenGL.so python record_scenario.py ${params} > ${log_file}
  python record_scenario.py ${params} > ${log_file}
done
