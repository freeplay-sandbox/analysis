#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace cv;
namespace po = boost::program_options;

int main(int argc, char **argv) {

    po::positional_options_description p;
    p.add("path", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produces help message")
        ("path", po::value<string>(), "path to the optical flow video (eg data/1234/videos/camera_purple_optical_flow.mkv)")
        ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
            .options(desc)
            .positional(p)
            .run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << argv[0] << "\n\n" << desc << "\n";
        return 1;
    }

    if (!vm.count("path")) {
        cerr << "You must provide a record path.\n";
        return 1;
    }



    VideoCapture cap(vm["path"].as<string>());
    if(!cap.isOpened()) {
        cerr << "Unable to open the video " << vm["path"].as<string>() << endl;
        return -1;
    }

    Mat bw;
    namedWindow("edges",1);

    cout << "frame, motion_intensity_avg, motion_intensity_stdev, motion_intensity_max, motion_direction_avg, motion_direction_stdev" << endl;

    int idx=0;
    for(;;)
    {
        Scalar magnitude_mean, magnitude_stddev, angle_mean, angle_stddev;
        double magnitude_min, magnitude_max;
        Mat frame;
        cap >> frame; // get a new frame from camera

        if (frame.empty()) break;

        Mat _hsv[3], hsv;
        cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        split(hsv, _hsv);

        // _hsv[0] -> angle
        // _hsv[2] -> magnitude


        cv::meanStdDev(_hsv[0], magnitude_mean, magnitude_stddev);
        cv::minMaxIdx(_hsv[0], &magnitude_min, &magnitude_max);

        cv::meanStdDev(_hsv[2], angle_mean, angle_stddev);

        cout << std::fixed << idx << "," << magnitude_mean[0] << "," << magnitude_stddev[0] << "," << magnitude_max << ",";
        cout << std::fixed << angle_mean[0] << "," << angle_stddev[0] << endl;

        idx++;
    }
    return 0;
}

