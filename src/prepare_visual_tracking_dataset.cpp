#include <fstream>
#include <iostream>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <string>
#include <thread> // std::this_thread
#include <vector>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/program_options.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <tf2_msgs/TFMessage.h>

#include "json.hpp"

#include "gaze_features.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace nlohmann; // json
namespace po = boost::program_options;


//const string BAG_FILE ("rectified_streams.bag");
const string BAG_FILE ("visual_tracking.bag");
const string POSES_FILE ("visual_tracking.poses.json");

typedef pair<ros::Time, pair<double, double>> TargetPose;
typedef map<ros::Time, pair<double, double>> TargetPoses;

bool interrupted = false;

void my_handler(int s){
    printf("Caught signal %d\n",s);
    interrupted = true; 
}

const double SANDTRAY_LENGTH = 0.6; //m
const double SANDTRAY_WIDTH = 0.338; //m

vector<double> normalizetarget(const pair<double, double> pos, bool mirror) {

    if (mirror)
        return {pos.first / SANDTRAY_LENGTH, (SANDTRAY_WIDTH + pos.second)/ SANDTRAY_WIDTH};
    else
        return {pos.first / SANDTRAY_LENGTH, -pos.second/ SANDTRAY_WIDTH};
}

pair<double, double> interpolate(const TargetPoses::value_type& a, const TargetPoses::value_type& b, const TargetPoses::key_type& t) {

    auto t1 = a.first.toSec();
    auto t2 = b.first.toSec();

    auto alpha = (t.toSec() - t1) / (t2-t1);

    auto x0 = a.second.first;
    auto dx = b.second.first - a.second.first;

    auto y0 = a.second.second;
    auto dy = b.second.second - a.second.second;

    return {x0 + alpha * dx, y0 + alpha * dy};
}


int main(int argc, char **argv) {


    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);

    po::positional_options_description p;
    p.add("path", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produces help message")
        ("version,v", "shows version and exits")
        ("path", po::value<string>(), "record path (must contain visual_tracking.{bag, poses.json})")
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
        cerr << "You must provide a record path.\n";
        return 1;
    }

    int total_nb_frames = 0;

    vector<string> topics {"camera_purple/rgb/image_raw/compressed",
                            "camera_yellow/rgb/image_raw/compressed"};

    json root;

    auto start = std::chrono::system_clock::now();
    cout << "Opening " << POSES_FILE << "..." << flush;
    std::ifstream file(vm["path"].as<string>() + "/" + POSES_FILE);
    file >> root;

    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;

    cout << "done (took " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << "s)" << endl << endl;

    for (auto topic : topics) {
        total_nb_frames += root[topic]["frames"].size();
        if (total_nb_frames == 0) {
            cerr << "Found no frames for topic " << topic << " in " << POSES_FILE << ". Aborting." << endl;
            exit(1);
        }
    }

    rosbag::Bag bag;
    rosbag::View view;

    cout << "Opening " << vm["path"].as<string>() << "/" << BAG_FILE << "..." << endl;
    bag.open(vm["path"].as<string>() + "/" + BAG_FILE, rosbag::bagmode::Read);
    view.addQuery(bag, rosbag::TopicQuery(vector<string>({"/tf"})));

    cout << total_nb_frames << " faces to align with TF data" << endl << endl;

    size_t foundTargets = 0;
    double delta = 0;

    double prev_x = 0, prev_y = 0;

    TargetPoses targetPoses;

    for(rosbag::MessageInstance const m : view)
    {

        auto tf = m.instantiate<tf2_msgs::TFMessage>();
        if (tf != nullptr) {

            for (const auto& t : tf->transforms) {
                if (t.child_frame_id == "visual_target") {
                    double x = t.transform.translation.x;
                    double y = t.transform.translation.y;

                    if (foundTargets > 0) {
                        delta += sqrt((x-prev_x) * (x-prev_x) + (y-prev_y) * (y-prev_y));
                        prev_x = x;
                        prev_y = y;
                    }

                    foundTargets++;

                    targetPoses[t.header.stamp] = {x, y};
                }
            }

        }


        if(interrupted) {
            cout << "Interrupted." << endl;
            break;
        }
    }

    json output;

    size_t idx = 0;

    bool mirror = false;

    for (auto topic : topics) {
        if(topic.find("yellow") != string::npos) mirror = true;

        for (auto frame : root[topic]["frames"]) {

            auto ts = ros::Time(frame["ts"].get<double>());

            if(frame["faces"].size() == 0) continue;

            if(frame["faces"].size() > 1 || frame["poses"].size() > 1) {
                cout << "At " << std::fixed << ts.toSec() << ", more than one face/skeleton (" << frame["faces"].size() << " faces, " << frame["poses"].size() << " skeletons). Skipping." << endl;
                continue;
            }


            bool found = false;

            // now we find the target's transforms just preceding and following this pose
            TargetPose prev;
            for(auto kv : targetPoses) {
                if (!prev.first.isZero() && prev.first < ts && kv.first > ts) {

                    auto target = interpolate(prev, kv, ts);

                    output[idx] = {getfeatures(frame, mirror), normalizetarget(target, mirror)};
                    idx++;
                    found = true;
                    break;
                }
                prev = kv;
            }

            if (!found) {

                cout << "Could not find suitable target poses! " << endl;
                cout << std::fixed << "Face pose ts: " << ts.toSec() << ", first target pose at " << (*targetPoses.begin()).first.toSec() << ", last one at " << (*targetPoses.rbegin()).first.toSec() << endl;
            }




        }
  }



    cout << "Found " << foundTargets << " targets; " << float(foundTargets) / (view.getEndTime() - view.getBeginTime()).toSec() << "Hz -- average delta between 2 poses: " << delta * 1000 / foundTargets << "mm" << endl;

    bag.close();

    std::ofstream fout(vm["path"].as<string>() + "/visual_tracking_dataset.json", ios_base::app);
    fout << output;
    fout.close();

    cout << "Aligned " << idx << " poses with the target. Saved as " << vm["path"].as<string>() << "/visual_tracking_dataset.json" << endl;
}
