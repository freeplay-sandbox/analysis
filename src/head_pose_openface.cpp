#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <tuple>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/image_processing/frontal_face_detector.h>

#include <OpenFace/LandmarkCoreIncludes.h>
#include <OpenFace/FaceAnalyser.h>
#include <OpenFace/GazeEstimation.h>
#include <OpenFace/Visualizer.h>

#include <boost/program_options.hpp>

#include <yaml-cpp/yaml.h>
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

const string YAML_EXPERIMENT_FILE ("experiment.yaml");
const string POSES_FILE ("freeplay.poses.json");
const string PURPLE_VIDEO_FILE ("videos/camera_purple_raw.mkv");
const string PURPLE_TOPIC ("camera_purple/rgb/image_raw/compressed");
const string YELLOW_VIDEO_FILE ("videos/camera_yellow_raw.mkv");
const string YELLOW_TOPIC ("camera_yellow/rgb/image_raw/compressed");


const size_t NB_LANDMARKS = 68; // discard the pupils (indices 68 and 69)

// these 2 heuristic helps with selecting the 'right' head if more than 1 face is visible.
// only the one whose height (in px) between the menton and the sellion is within these
// bounds is kept.
const int MIN_HEAD_HEIGHT = 50;
const int MAX_HEAD_HEIGHT = 150;

/**
 * Returns the facial landmarks previously detected with OpenPose as a list [x1, x2...,xn,y1...yn]
 * of 2D image coordinates, followed by the left pupil and the right pupil,
 * follow by the average of landmark detection confidence.
 */
tuple<Mat_<double>, Point2f, Point2f, float> readFaceLandmarks(cv::Size
        image_size, const json& face) {

    Mat_<double> landmarks(NB_LANDMARKS * 2, 1);
    Point2f l_pupil;
    Point2f r_pupil;

    if (face.is_null()) return {landmarks, l_pupil, r_pupil, 0};

    auto w = image_size.width;
    auto h = image_size.height;

    auto nb_faces = face.size();
    float total_confidence;

    for (size_t face_idx = 1; face_idx <= nb_faces; face_idx++) { // face indicies start at 1!

        total_confidence = 0;

        for(size_t idx = 0; idx < NB_LANDMARKS; idx++) {

            float confidence = face[to_string(face_idx)][idx][2].get<float>();
            total_confidence += confidence;

            auto x = w*face[to_string(face_idx)][idx][0].get<float>();
            auto y = h*face[to_string(face_idx)][idx][1].get<float>();

            landmarks.at<double>(idx) = x;
            landmarks.at<double>(idx + NB_LANDMARKS) = y;
        }

        auto x = w*face[to_string(face_idx)][68][0].get<float>();
        auto y = h*face[to_string(face_idx)][68][1].get<float>();
        l_pupil = Point2f(x, y);

        x = w*face[to_string(face_idx)][69][0].get<float>();
        y = h*face[to_string(face_idx)][69][1].get<float>();
        r_pupil = Point2f(x, y);

        // face_height is the distance (in px) between the menton and the sellion
        Point2f menton(landmarks.at<double>(8), landmarks.at<double>(8 + NB_LANDMARKS));
        Point2f sellion(landmarks.at<double>(27), landmarks.at<double>(27 + NB_LANDMARKS));
        auto face_height = cv::norm(menton-sellion);

        // if we have more than one face, and the current face matches our head size heuristic,
        // use that
        if (nb_faces > 1 && (face_height > MIN_HEAD_HEIGHT && face_height < MAX_HEAD_HEIGHT)) break;
    }


    return {landmarks, l_pupil, r_pupil, total_confidence/NB_LANDMARKS};
}

void write_csv_header(std::ofstream output_file) {

    if (!output_file.is_open())
    {
        std::cout << "The output CSV file is not open, exiting" << std::endl;
        exit(1);
    }

    output_file << "id, timestamp,";
    output_file << "purple_confidence, "
                   "purple_pose.x, purple_pose.y, purple_pose.z, purple_pose.rx, purple_pose.ry, purple_pose.rz, "
                   "purple_gaze.x, purple_gaze.y, purple_gaze.z, "
                   "purple_AU01, purple_AU02, purple_AU04, purple_AU05, purple_AU06, purple_AU07, "
                   "purple_AU09, purple_AU10, purple_AU12, purple_AU14, purple_AU15, purple_AU17, "
                   "purple_AU20, purple_AU23, purple_AU25, purple_AU26, purple_AU45";

    output_file << "yellow_confidence, "
                   "yellow_pose.x, yellow_pose.y, yellow_pose.z, yellow_pose.rx, yellow_pose.ry, yellow_pose.rz, "
                   "yellow_gaze.x, yellow_gaze.y, yellow_gaze.z, "
                   "yellow_AU01, yellow_AU02, yellow_AU04, yellow_AU05, yellow_AU06, yellow_AU07, "
                   "yellow_AU09, yellow_AU10, yellow_AU12, yellow_AU14, yellow_AU15, yellow_AU17, "
                   "yellow_AU20, yellow_AU23, yellow_AU25, yellow_AU26, yellow_AU45";

    output_file << endl;
}

void write_csv_line(std::ofstream output_file,
                    int frame_num, long timestamp, double confidence, 
                    cv::Vec6d& pose_estimate,
                    const cv::Point3f& gazeDirection,
                    const std::vector<std::pair<std::string, double> >& au_intensities,
                    const std::vector<std::pair<std::string, double> >& au_occurences) {

    if (!output_file.is_open())
    {
        std::cout << "The output CSV file is not open, exiting" << std::endl;
        exit(1);
    }

    // Making sure fixed and not scientific notation is used
    output_file << std::fixed;
    output_file << std::noshowpoint;

    output_file << frame_num << ", " << timestamp;
    output_file << std::setprecision(2);
    output_file << ", " << confidence;

    // pose
    output_file << std::setprecision(1);
    output_file << ", " << pose_estimate[0] << ", " << pose_estimate[1] << ", " << pose_estimate[2];
    output_file << std::setprecision(3);
    output_file << ", " << pose_estimate[3] << ", " << pose_estimate[4] << ", " << pose_estimate[5];

    // gaze
    output_file << std::setprecision(3);
    output_file << ", " << gazeDirection.x << ", " << gazeDirection.y << ", " << gazeDirection.z;

    // Action Units
    
    output_file << std::setprecision(1);
    std::map<std::string, std::pair<bool, double>> aus;

    // first, prepare a mapping "AU name" -> { present, intensity }
    for (size_t idx = 0; idx < au_intensities.size(); idx++) {
        aus[au_intensities[idx].first] = std::make_pair(au_occurences[idx].second != 0, au_intensities[idx].second);
    }

    for (auto& au : aus) {
        bool present = au.second.first;
        double intensity = au.second.second;
        if (present) {
            output_file << ", " << intensity;
        }
        else {
            output_file << ", 0";
        }
    }
    output_file << std::endl;
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
        cerr << "You must provide a path to the experiment to process.\n";
        return 1;
    }


    YAML::Node experiment = YAML::LoadFile(vm["path"].as<string>() + "/" + YAML_EXPERIMENT_FILE);

    bool childchild = (experiment["condition"].as<string>() == "childchild");

    if (!childchild) {
        cout << "Child-robot experiment. Skipping it for now." << endl;
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

    // Find the first common timestamp
    size_t purple_frame_start_idx = 0;
    size_t yellow_frame_start_idx = 0;

    double start_timestamp = 0;
    bool found_start_timestamp = false;

    for (const auto& f_p : poses[PURPLE_TOPIC]["frames"]) {
        auto ts_p = f_p["ts"].get<double>();
        for (const auto& f_y : poses[YELLOW_TOPIC]["frames"]) {
            auto ts_y = f_y["ts"].get<double>();

            if (ts_y > ts_p) {break;}
            if (ts_y == ts_p) {
                start_timestamp = ts_y;
                found_start_timestamp = true;
                break;
            }

            yellow_frame_start_idx += 1;
        }
        
        if(found_start_timestamp) {break;}
        purple_frame_start_idx += 1;
    }

    if (!found_start_timestamp) {
        cout << "Impossible to find a suitable common timestamp for the 2 video streams! Aborting" << endl;
        return -1;
    }

    cout << "Found a common start timestamp (" << start_timestamp << "). Purple stream must skip " << purple_frame_start_idx << " frames, yellow stream " << yellow_frame_start_idx << "frames." <<endl;

    VideoCapture purple_capture(vm["path"].as<string>() + "/" + PURPLE_VIDEO_FILE);
    VideoCapture yellow_capture(vm["path"].as<string>() + "/" + YELLOW_VIDEO_FILE);

    // skipping needed frames to align the 2 video streams
    Mat frame;
    for (size_t i; i < purple_frame_start_idx; i++) {
        purple_capture >> frame;
    }
    for (size_t i; i < yellow_frame_start_idx; i++) {
        yellow_capture >> frame;
    }


    auto nb_purple_frames = poses[PURPLE_TOPIC]["frames"].size() - purple_frame_start_idx;
    auto nb_yellow_frames = poses[YELLOW_TOPIC]["frames"].size() - yellow_frame_start_idx;

    if (nb_purple_frames != nb_yellow_frames) {
        cout << "The two streams do not end at the same time (delta of " << abs(nb_purple_frames-nb_yellow_frames)/30. << " sec). I'll stop after the shortest one." << endl;
    }

    auto nb_frames_video = min(nb_purple_frames, nb_yellow_frames);

    cout << nb_frames_video << " frames to process." << endl;



    Utilities::Visualizer visualizer(true, false, false);
    // Load the models if images found
    LandmarkDetector::FaceModelParameters det_parameters;

    // The modules that are being used for tracking
    cout << "\n\n######################\nLoading the models from " << det_parameters.model_location << endl;
    LandmarkDetector::CLNF face_model(det_parameters.model_location);

    // Load facial feature extractor and AU analyser (make sure it is static)
    FaceAnalysis::FaceAnalyserParameters face_analysis_params;
    face_analysis_params.OptimizeForImages();
    FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

    bool model_initialized = false;

    start = std::chrono::system_clock::now();

    cout << "\n\nStarting gaze processing\n\n" << endl;

    double timestamp;

    size_t frame_idx = purple_frame_start_idx;

    for(;;)
    {
        timestamp = poses[PURPLE_TOPIC]["frames"][frame_idx]["ts"].get<double>();

        Mat frame;
        purple_capture >> frame; 
        frame_idx += 1;

        if (frame.empty()) {
            cout << "End of video."<< endl;
            break;
        }

        int percent = frame_idx * 100 / nb_frames_video;
        auto intermediate = std::chrono::system_clock::now();
        auto fps = frame_idx / ((double)std::chrono::duration_cast<std::chrono::milliseconds>(intermediate-start).count() * 1e-3);

        cout << "\x1b[FDone " << percent << "% (" << frame_idx << "/" << (int)nb_frames_video << " frames, " << std::fixed << std::setprecision(1) << fps << " fps)" << endl;

        Mat grayscale_image;
        cvtColor(frame, grayscale_image, CV_BGR2GRAY);

        if(!model_initialized) {
            bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, face_model, det_parameters);
            if(success) {
                cout << "Face model initialized. Continuing with pre-detected OpenPose facial landmarks" << endl << endl;
                model_initialized = true;
            }
            else {
                continue;
            }
        }


        // read landmarks from pre-recorded OpenPose JSON file
        Mat_<double> landmarks;
        Point2f l_pupil;
        Point2f r_pupil;
        float confidence;

        std::tie(landmarks, l_pupil, r_pupil, confidence) = readFaceLandmarks(frame.size(), poses[PURPLE_TOPIC]["frames"][frame_idx]["faces"]);
        face_model.detected_landmarks = landmarks;
        face_model.detection_certainty = confidence;

        // overwrite OpenFace's eye detector landmark.
        // This is a hack: unlike OpenFace, OpenPose does detect directly the pupil.
        // OpenFace replace pupil detection by averaging the eye contour.
        // We trick OpenFace by providing an eye contour made exclusively of the pupil
        // position.
        Mat_<double> l_eye(28*2,1), r_eye(28*2,1);
        for (size_t i = 0; i < 28; i++) {
            l_eye.at<double>(i) = l_pupil.x;
            r_eye.at<double>(i) = r_pupil.x;
            l_eye.at<double>(i+28) = l_pupil.y;
            r_eye.at<double>(i+28) = r_pupil.y;
        }
        face_model.hierarchical_models[1].detected_landmarks = l_eye; // left eye
        face_model.hierarchical_models[1].detection_certainty = confidence;
        face_model.hierarchical_models[2].detected_landmarks = r_eye; // left eye
        face_model.hierarchical_models[2].detection_certainty = confidence;

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

        // Aligned face
        //cv::Mat sim_warped_img;
        //face_analyser.AddNextFrame(frame, face_model.detected_landmarks, face_model.detection_success, timestamp);
        //face_analyser.GetLatestAlignedFace(sim_warped_img);
        //imshow("Aligned face", sim_warped_img);

        //visualizer.SetImage(frame, FX, FY, CX, CY);
        //visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
        //visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
        //visualizer.SetObservationGaze(gaze_direction0, gaze_direction1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, FX, FY, CX, CY), face_model.detection_certainty);
        //visualizer.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());

        //imshow("Gaze", visualizer.GetVisImage());
        //waitKey(10);

}

    return 0;
}

