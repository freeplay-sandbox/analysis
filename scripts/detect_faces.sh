#! /bin/bash

EXEC=$HOME/src/freeplay_sandbox_analysis/build/devel/lib/freeplay_sandbox_analysis/faces_analysis
DATASET_ROOT=/media/slemaignan/DoRoThy1/freeplay_sandox/data
NB_CPUS=7

DLIB_MODEL=$HOME/ros-dev/share/gazr/shape_predictor_68_face_landmarks.dat

orig_path=$(pwd)

cd $DATASET_ROOT
paths=(2017**/experiment.yaml)


for ((i=0;i<${#paths[@]};i+=$NB_CPUS))
do
    for ((j=0;j<$NB_CPUS;j++))
    do
        if [[ $i+$j -ge ${#paths[@]} ]]; then break; fi 
        f=${paths[$i+$j]}


        dir=$(dirname $f)
        bag=$(basename $f)

        if [ ! -f $dir/faces.yaml ]; then

            echo "Detecting faces for experiment $f..."
            # the brackets mean 'start in a subshell' and we put that subshell in the background
            ( $EXEC --topic="camera_purple/rgb/image_raw/compressed" --model=$DLIB_MODEL $dir ; $EXEC --topic="camera_yellow/rgb/image_raw/compressed" --model=$DLIB_MODEL $dir ) &

        else
            echo "faces.yaml already exist. Skipping."
        fi
    done

    echo "Waiting for the first batch to complete"
    for job in `jobs -p`
    do
        echo "Job $job..."
        wait $job
    done
    echo "Starting next batch"
done

cd $orig_path
echo "Done"
