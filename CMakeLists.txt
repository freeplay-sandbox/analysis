cmake_minimum_required(VERSION 2.8.3)
project(freeplay_sandbox_analysis)

set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "2")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

add_definitions(-std=c++11 -DFREEPLAY_ANALYSIS_VERSION=${VERSION})

find_package(OpenCV 3.1 COMPONENTS core imgproc calib3d highgui REQUIRED cudaoptflow)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

if(OpenCV_CUDA_VERSION)
    option(WITH_CUDA_OPENCV_SUPPORT "Build with CUDA OpenCV support" ON)
    if(${WITH_CUDA_OPENCV_SUPPORT})
        message(STATUS "OpenCV compiled with CUDA v${OpenCV_CUDA_VERSION} support. Great. Using it for optical flow.")
        add_definitions(-DCUDA_CV)
    else()
        message(STATUS "Even though OpenCV compiled with CUDA, support manually disabled.")
    endif()
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options)

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

find_package(Caffe)
include_directories(${Caffe_DIR}/../../include)

option(WITH_CAFFE_SUPPORT "Build with Caffe support" ${Caffe_FOUND})

if(${WITH_CAFFE_SUPPORT} AND ${Caffe_FOUND})
    message(STATUS "Found Caffe at ${Caffe_DIR} -- building with support for gaze estimation")
    add_definitions(-DWITH_CAFFE)
    option(WITH_OPENPOSE "Build with OpenPose support" OFF)
else()
    message(STATUS "Caffe not enabled/not found -- pose detection will not be available")
endif()


if(${WITH_OPENPOSE})
    include_directories(${Caffe_INCLUDE_DIRS})

    find_package(PkgConfig REQUIRED)
    pkg_search_module(OPENPOSE REQUIRED openpose)
    include_directories(${OPENPOSE_INCLUDE_DIRS})
    link_directories(${OPENPOSE_LIBRARY_DIRS})
endif()

find_package(catkin REQUIRED COMPONENTS 
    roscpp
    rosbag
    tf2
    std_msgs
    visualization_msgs
    message_filters
    sensor_msgs
    cv_bridge
    image_geometry
    audio_common_msgs
    )

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
    CATKIN_DEPENDS rosbag
    LIBRARIES 
)

#add_executable(mutual_gaze_segmentation src/mutual_gaze_segmentation.cpp)
#target_link_libraries(mutual_gaze_segmentation ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

if(${WITH_OPENPOSE})
    add_executable(extract_poses src/extract_poses.cpp)
    target_link_libraries(extract_poses ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES} ${OPENPOSE_LIBRARIES} ${catkin_LIBRARIES})

    install(TARGETS extract_poses
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endif()

add_executable(pose_analysis src/pose_analysis.cpp)
target_link_libraries(pose_analysis ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

add_executable(face_detection src/face_detection.cpp)
target_link_libraries(face_detection ${Boost_LIBRARIES})


add_executable(check_pose_data src/check_pose_data.cpp)
target_link_libraries(check_pose_data ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES})


add_executable(replay_with_poses src/replay_with_poses.cpp src/gaze_features.cpp src/head_pose_estimator.cpp)
target_link_libraries(replay_with_poses ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

add_executable(prepare_visual_tracking_dataset src/prepare_visual_tracking_dataset.cpp src/gaze_features.cpp)
target_link_libraries(prepare_visual_tracking_dataset ${catkin_LIBRARIES})

#install(TARGETS mutual_gaze_segmentation extract_poses
install(TARGETS replay_with_poses
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(optical_flow src/optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

add_executable(vad src/vad.cpp)
target_link_libraries(vad ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})


catkin_install_python(PROGRAMS
    nodes/analysis
    nodes/tf_camera_replicator
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
    launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

