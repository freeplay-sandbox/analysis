Free-play Sandbox -- Data-analyse GUI
=====================================

*This is the sister repository to [the ROS-based
framework](https://github.com/severin-lemaignan/freeplay-sandbox-ros) and [the
QtQuick-based GUI](https://github.com/severin-lemaignan/freeplay-sandbox-qt) of
the 'Free-play Sandbox' experimental framework for Cognitive Human-Robot
Interaction research.*

![Screenshoot of the GUI](docs/screenshot.jpg)

A [rqt](http://wiki.ros.org/rqt) GUI to visualise and analyse the 'freeplay
sandbox' recordings.

The interface is heavily based on
[rqt_bag](https://github.com/ros-visualization/rqt_common_plugins/tree/master/rqt_bag).


Other processings
-----------------

- to extract a video stream from one of the bag file and save it as a video:

```
$ rosrun bag_tools make_video.py <topic> <bag> --output <output.mp4> --fps <fps>
```

For instance:
```
$ rosrun bag_tools make_video.py env_camera/qhd/image_color/compressed freeplay.bag --output freeplay_env.mp4 --fps 28.0
```

(note that, due to an upstream bug, one needs first to replace the type of the `fps`
parameter from `int` to `float` in `make_video.py` for non-integer FPS to work)

- to extract audio and save it as an audio file:

```
$ rosrun audio_play audio_play audio:=<topic> _dst:=<file.ogg>
```

For instance:
```
$ rosrun audio_play audio_play audio:=camera_purple/audio _dst:=freeplay_purple.ogg
```
