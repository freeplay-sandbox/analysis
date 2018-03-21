#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>

#include <OpenFace/LandmarkCoreIncludes.h>
#include <OpenFace/FaceAnalyser.h>
#include <OpenFace/GazeEstimation.h>
#include <OpenFace/Visualizer.h>

#include <boost/program_options.hpp>

#include "json.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace nlohmann; // json

// focal lengths
const double FX = 697.811; // extracted from actual ROS bag records
const double FY = 697.811; // extracted from actual ROS bag records

// optical center
const double CX = 479.047 ; // extracted from actual ROS bag records
const double CY = 261.227; // extracted from actual ROS bag records

const string POSES_FILE ("freeplay.poses.json");
const string VIDEO_FILE ("videos/camera_purple_raw.mkv");
const string TOPIC ("camera_purple/rgb/image_raw/compressed");

const size_t NB_LANDMARKS = 68; // discard the pupils (indices 68 and 69)

/**
 * Returns the facial landmarks previously detected with OpenPose as a list [x1, x2...,xn,y1...yn]
 * of 2D image coordinates
 */
Mat_<double> readFaceLandmarks(cv::Size image_size, const json& face) {

    Mat_<double> landmarks(NB_LANDMARKS * 2, 1);

    if (face.is_null()) return landmarks;

    auto w = image_size.width;
    auto h = image_size.height;

    // TODO: if more than one face, select the best one (closest to center or right size)
    int face_idx = 1;

    for(size_t idx = 0; idx < NB_LANDMARKS; idx++) {

        float confidence = face[to_string(face_idx)][idx][2].get<float>();

        auto x = w*face[to_string(face_idx)][idx][0].get<float>();
        auto y = h*face[to_string(face_idx)][idx][1].get<float>();

        landmarks.at<double>(idx) = x;
        landmarks.at<double>(idx + NB_LANDMARKS) = y;
    }

    return landmarks;
}

int main (int argc, char **argv)
{

    po::positional_options_description p;
    p.add("path", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produces help message")
        ("version,v", "shows version and exits")
//        ("models", po::value<string>()->default_value("models/"), "path to OpenPose models")
//        ("bag", po::value<string>()->default_value("freeplay"), "Bag file, without the '.bag' extension")
        ("path", po::value<string>(), "path to the source video")
//        ("face", po::value<bool>()->default_value(true), "detect faces as well")
//        ("hand", po::value<bool>()->default_value(true), "detect hands as well")
        ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
            .options(desc)
            .positional(p)
            .run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << argv[0] << " " << STR(FREEPLAY_ANALYSIS_VERSION) << "\n\n" << desc << "\n";
        return 1;
    }

    if (vm.count("version")) {
        cout << argv[0] << " " << STR(FREEPLAY_ANALYSIS_VERSION) << "\n";
        return 0;
    }

    if (!vm.count("path")) {
        cerr << "You must provide a path to the video to process.\n";
        return 1;
    }

    //////////////////// LOAD poses.json  //////////////////////
    json poses;

    auto start = std::chrono::system_clock::now();
    cerr << "Opening " << POSES_FILE << "..." << flush;
    std::ifstream file(vm["path"].as<string>() + "/" + POSES_FILE);
    file >> poses;

    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;

    cerr << "done (took " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << "s)" << endl << endl;
    /////////////////////////////////////////////////////////////




    Utilities::Visualizer visualizer(true, false, false);

    VideoCapture capture(vm["path"].as<string>() + "/" + VIDEO_FILE);
    auto nb_frames_video = capture.get(CV_CAP_PROP_FRAME_COUNT);


    // Load the models if images found
    LandmarkDetector::FaceModelParameters det_parameters;

    // The modules that are being used for tracking
    cout << "Loading the model " << det_parameters.model_location << endl;
    LandmarkDetector::CLNF face_model(det_parameters.model_location);

    // Load facial feature extractor and AU analyser (make sure it is static)
    FaceAnalysis::FaceAnalyserParameters face_analysis_params;
    face_analysis_params.OptimizeForImages();
    FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

    bool model_initialized = false;

    size_t frame_idx;
    start = std::chrono::system_clock::now();

    cout << "\n\nStarting gaze processing\n\n" << endl;
    double timestamp = 0;

    for(;;)
    {
        Mat frame;
        capture >> frame; 
        frame_idx += 1;
        timestamp += 1/30.;

        if (frame.empty()) {
            cout << "End of video."<< endl;
            break;
        }

        int percent = frame_idx * 100 / nb_frames_video;
        auto intermediate = std::chrono::system_clock::now();
        auto fps = frame_idx / ((double)std::chrono::duration_cast<std::chrono::milliseconds>(intermediate-start).count() * 1e-3);

        cout << "\x1b[FDone " << percent << "% (" << frame_idx << "/" << (int)nb_frames_video << " frames, " << std::fixed << std::setprecision(1) << fps << " fps)" << endl;


        
        Mat grayscale_image;
        // Making sure the image is in uchar grayscale
        //cv::Mat_<uchar> grayscale_image = image_reader.GetGrayFrame();
        cvtColor(frame, grayscale_image, CV_BGR2GRAY);

        if(!model_initialized) {
            bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, face_model, det_parameters);
            if(success) {
                cout << "Face model initialized. Continuing with pre-detected OpenPose facial landmarks" << endl;
                model_initialized = true;
            }
            else {
                continue;
            }
        }
        //bool success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);
        //if (!success) {
        //    cout << "\t\t\t\t\t\t\t\t\tFailed to detect landmark on frame " << frame_idx << endl;
        //    continue;
        //}

        // read landmarks from pre-recorded OpenPose JSON file
        auto landmarks = readFaceLandmarks(frame.size(), poses[TOPIC]["frames"][frame_idx]["faces"]);
        face_model.detected_landmarks = landmarks;
        face_model.detection_certainty = 1.0;


        // Estimate head pose and eye gaze
        Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, FX, FY, CX, CY);

        // Gaze tracking, absolute gaze direction
        Point3f gaze_direction0(0, 0, 0);
        Point3f gaze_direction1(0, 0, 0);
        Vec2d gaze_angle(0, 0);

        if (face_model.eye_model)
        {
            GazeAnalysis::EstimateGaze(face_model, gaze_direction0, FX, FY, CX, CY, true);
            GazeAnalysis::EstimateGaze(face_model, gaze_direction1, FX, FY, CX, CY, false);
            gaze_angle = GazeAnalysis::GetGazeAngle(gaze_direction0, gaze_direction1);
        }
        
        //cv::Mat sim_warped_img;
        //face_analyser.AddNextFrame(frame, face_model.detected_landmarks, face_model.detection_success, timestamp);

        //face_analyser.GetLatestAlignedFace(sim_warped_img);

        visualizer.SetImage(frame, FX, FY, CX, CY);
        //visualizer.SetObservationFaceAlign(sim_warped_img);
        visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
        visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
        visualizer.SetObservationGaze(gaze_direction0, gaze_direction1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, FX, FY, CX, CY), face_model.detection_certainty);
        //visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());

        // detect key presses
        //char character_press = visualizer.ShowObservation();

        // quit processing the current sequence (useful when in Webcam mode)
        //if (character_press == 'q')
        //{
        //    break;
        //}
        imshow("toto", visualizer.GetVisImage());
        waitKey(10);

}

    return 0;
}

