cmake_minimum_required(VERSION 2.8.3)
project(freeplay_sandbox_analysis)

set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

add_definitions(-std=c++11 -DFREEPLAY_ANALYSIS_VERSION=${VERSION})

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS 
    roscpp
    rosbag
    tf
    std_msgs
    visualization_msgs
    sensor_msgs
    cv_bridge
    gazr
    image_geometry
    )

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
    CATKIN_DEPENDS rosbag
    LIBRARIES 
)

find_package(OpenCV COMPONENTS core imgproc calib3d highgui REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
if(${OpenCV_VERSION} VERSION_GREATER 2.9.0)
    set(OPENCV3 TRUE)
    add_definitions(-DOPENCV3)
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mutual_gaze_segmentation src/mutual_gaze_segmentation.cpp)
target_link_libraries(mutual_gaze_segmentation ${dlib_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

add_executable(faces_analysis src/faces_analysis.cpp)
target_link_libraries(faces_analysis ${dlib_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES})

install(TARGETS mutual_gaze_segmentation faces_analysis
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
    nodes/analysis
    nodes/dataset_stats
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

