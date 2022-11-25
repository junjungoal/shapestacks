#!/bin/sh
echo "Recording ShapeStacks scenarios for MuJoCo."

# filter parameters
# $1: dataset name
# $2: height filter

# paths
DATASET_NAME=$1
DATASET_ROOT_DIR="${SHAPESTACKS_CODE_HOME}/data/${DATASET_NAME}"
MJCF_ROOT_DIR="${DATASET_ROOT_DIR}/mjcf"
RECORD_ROOT_DIR="${DATASET_ROOT_DIR}/recordings"

# file filter
FILTER="h=[$2]"

# recording options
TIME=1
FPS=8
MAX_FRAMES=1
RES="224 224"
CAMERAS="cam_1 cam_2 cam_3"
FORMAT="depth"

# helper functions
create_params()
{
  model_path=$1   # $1: model_path
  record_path=$2  # $2: record_path

  echo "--mjmodel_path ${model_path} --record_path ${record_path} \
      --mjsim_time ${TIME} --fps ${FPS} --max_frames ${MAX_FRAMES} \
      --res ${RES} --cameras ${CAMERAS} \
      --formats ${FORMAT}"
}

###
# Main body
###

# directory setup
# mkdir ${DATASET_ROOT_DIR}
# mkdir ${RECORD_ROOT_DIR}

# main loop over all simulation environments to record
for env_file in `ls ${MJCF_ROOT_DIR} | grep env_ | grep ${FILTER}`; do
  date
  echo "Recording ${env_file%".xml"} ..."

  # set up directory
  record_dir=${RECORD_ROOT_DIR}/${env_file%".xml"}
  log_file=${record_dir}/depth_log.txt
  mkdir ${record_dir}

  # create params and render
  params=$(create_params "${MJCF_ROOT_DIR}/$env_file" $record_dir)
  # echo $params
  # LD_PRELOAD=/usr/lib/nvidia-384/libOpenGL.so python3 record_scenario.py ${params} > ${log_file}
  LD_PRELOAD=/usr/lib/nvidia-418/libOpenGL.so python3 record_scenario.py ${params} > ${log_file}
done
