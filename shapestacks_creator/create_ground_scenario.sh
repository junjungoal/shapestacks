#!/bin/sh
echo "Recording ShapeStacks scenarios from MuJoCo."

# filter parameters
# $1: dataset name
# $2: height filter

# paths


# recording options
# TIME=4
MAX_HEIGHT=4
COLOR_MODE="original"

# helper functions
create_params()
{
  v=$1
  template_path=${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/mjcf   # $1: model_path
  export_path=${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/mjcf  # $2: record_path

  # randomize light and textures
  height=`python -c "import random; print(random.randint(3,${MAX_HEIGHT}))"`
  vcom=`python -c "import random; print(random.randint(0,${height}))"`
  shape_num=`python -c "import random; print(random.randint(1,3))"`
  shapes=`python -c "import numpy as np; [print(name) for name in np.random.choice(['cuboid', 'cylinder', 'sphere'], size=${shape_num}, replace=False)]"`
  # echo $lightid $walltex $floortex
  MJMODEL_NAME="h=${height}-vcom=${vcom}-vpsf=0-v=${v}"

  echo "--mjmodel_name ${MJMODEL_NAME}  --template_path ${template_path} --export_path ${export_path} \
      --shapes ${shapes} \
      --height ${height} \
      --vcom ${vcom} \
      --unique_colors"
}


###
# Main body
###

# directory setup
# mkdir ${DATASET_ROOT_DIR}
# mkdir ${RECORD_ROOT_DIR}

# main loop over all simulation environments to record

i=0
while [ $i -ne 6000 ]
do
        params=$(create_params $i)
        python create_small_ground_scenario.py ${params}
        i=$(($i+1))
        echo "File $i"
done

