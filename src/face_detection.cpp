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

#include <boost/program_options.hpp>

#include "json.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


using namespace std;
using namespace nlohmann; // json
namespace po = boost::program_options;

const string POSES_FILE ("freeplay.poses.json");
bool interrupted = false;

void my_handler(int s){
    printf("Caught signal %d\n",s);
    interrupted = true; 
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
        ("path", po::value<string>(), "record path (must contain experiment.yaml and freeplay.bag)")
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


    json root;

    auto start = std::chrono::system_clock::now();
    //cerr << "Opening " << POSES_FILE << "..." << flush;
    std::ifstream file(vm["path"].as<string>() + "/" + POSES_FILE);
    file >> root;

    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;

    //cerr << "done (took " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << "s)" << endl << endl;

    auto frames = root["camera_purple/rgb/image_raw/compressed"]["frames"];
    uint nb_frames = frames.size();

    //cout << "Total frames: " << nb_frames << endl;

    uint nb_faces = 0;
    uint nb_duplicate_faces = 0;

    for(auto frame : frames) {
        if(frame.count("faces") > 0) nb_faces++;
        if(frame.count("faces") > 1) nb_duplicate_faces++;
    }
    //cout << "Total frames with faces: " << nb_faces << endl;
    cout << vm["path"].as<string>() + "/" + POSES_FILE << ", " << nb_frames << ", " << nb_faces << ", " << nb_duplicate_faces << endl;

}
