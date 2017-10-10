#! /bin/bash

EXEC=$HOME/src/freeplay-sandbox-analysis/build/devel/lib/freeplay_sandbox_analysis/faces_detection
DATASET_ROOT=/media/$USER/DoRoThy1/freeplay_sandbox/data

orig_path=$(pwd)

cd $DATASET_ROOT
paths=(2017**/experiment.yaml)


for ((i=0;i<${#paths[@]};i++))
do
    f=${paths[$i]}
    
    dir=$(dirname $f)
    
    if [ -f $dir/freeplay.poses.json ]; then
    
        echo "Detecting faces for experiment $f (experiment $((i+1))/${#paths[@]}..."
        $EXEC --path $dir
    
    fi
done

cd $orig_path
echo "Done"
