#! /bin/bash

EXEC=$HOME/src/freeplay-sandbox-analysis/build/devel/lib/freeplay_sandbox_analysis/extract_poses
DATASET_ROOT=/media/$USER/DoRoThy1/freeplay_sandbox/data

OPENPOSE_MODELS=/opt/openpose/models

orig_path=$(pwd)

cd $DATASET_ROOT
paths=(2017**/experiment.yaml)


for ((i=0;i<${#paths[@]};i++))
do
    f=${paths[$i]}
    
    dir=$(dirname $f)
    
    if [ ! -f $dir/poses.json ]; then
    
        echo "Detecting faces for experiment $f (experiment $((i+1))/${#paths[@]}..."
        $EXEC --model=$OPENPOSE_MODELS $dir
    
    else
        echo "poses.json already exist. Skipping $dir."
    fi
done

cd $orig_path
echo "Done"
