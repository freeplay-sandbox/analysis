#! /bin/bash

EXEC=$HOME/src/freeplay-sandbox-analysis/build/devel/lib/freeplay_sandbox_analysis/prepare_visual_tracking_dataset
DATASET_ROOT=/media/$USER/PInSoRo-backup/freeplay_sandbox/data

orig_path=$(pwd)

cd $DATASET_ROOT
paths=(2017**/visual_tracking.poses.json)


for ((i=0;i<${#paths[@]};i++))
do
    f=${paths[$i]}
    
    dir=$(dirname $f)
    
    if [ ! -f $dir/visual_tracking_dataset.json ]; then
    
        echo "Preparing visual_tracking dataset for experiment $f (experiment $((i+1))/${#paths[@]})..."
        $EXEC --path=$dir  || { echo 'detection failed' ; exit 1; }

    
    else
        echo "visual_tracking_dataset.json already exist. Skipping $dir."
    fi
done

cd $orig_path
echo "Done"
