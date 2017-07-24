#! /bin/bash

EXEC=$HOME/src/freeplay-sandbox-analysis/build/devel/lib/freeplay_sandbox_analysis/faces_analysis
DATASET_ROOT=/media/$USER/DoRoThy1/freeplay_sandbox/data

OPENPOSE_MODELS=/opt/openpose/models

orig_path=$(pwd)

cd $DATASET_ROOT
paths=(2017**/experiment.yaml)


for ((i=0;i<${#paths[@]};i++))
do
    f=${paths[$i]}
    
    dir=$(dirname $f)
    
    if [ ! -f $dir/visual_tracking.poses.json ]; then
    
        echo "Detecting faces for visual tracking task $f (experiment $((i+1))/${#paths[@]})..."
        $EXEC --model=$OPENPOSE_MODELS --bag visual_tracking $dir  || { echo 'detection failed' ; exit 1; }
    
    else
        echo "visual_tracking.poses.json already exist. Skipping $dir."
    fi
done

cd $orig_path
echo "Done"
