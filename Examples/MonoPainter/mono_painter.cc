/**
* This file is a modified version of Dyna-SLAM.<https://github.com/BertaBescos/DynaSLAM>
*
* This file is part of DynaSLAM_MonoPainter.
* For more information see <https://github.com/Ocenal/DynaSLAM_MonoPainter>.
*
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include<opencv2/core/core.hpp>

#include "Geometry.h"
#include "MaskNet.h"

#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4 && argc != 5 && argc != 6)
    {	//0	EXECUTION
    	//1	vocabulary
    	//2	camera intrinsic parameter
    	//3	image sequence
    	//-4	masks
    	//-5	output
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence (path_to_masks) (path_to_output)" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Initialize Mask R-CNN
    DynaSLAM::SegmentDynObject* MaskNet;
    if (argc==5 || argc==6)
    {
        cout << "Loading Mask R-CNN. This could take a while..." << endl;
        MaskNet = new DynaSLAM::SegmentDynObject();
        cout << "Mask R-CNN loaded!" << endl;
    }

	//A vector for frame outputing
	std::vector<cv::Mat> imgRGBOut;
	//The argument number is constrained to be 6 for MonoPainter
	if (argc==6)
	{
		cerr<<"\nTry to initialize Special ORB-SLAM2 System for MonoPainter..."<<endl;
	}else{
		cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence path_to_masks path_to_output" << endl;
		return -1;
	}
	
	// Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(imgRGBOut,argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);
	cerr<<"\nSpecial ORB-SLAM2 System for MonoPainter initialized successfully!"<<endl;
    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;


	//create the dir for painted frames' storage
	if (argc==6)
    {
        std::string dir = string(argv[5]);
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[5]) + "/PaintedFrames/";
		mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
	
    // Main loop
    cv::Mat im;

    // Dilation settings
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );

	
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Segment out the images
        cv::Mat mask = cv::Mat::ones(480,640,CV_8U);
        if(argc == 5 || argc==6)
        {
            cv::Mat maskRCNN;
            maskRCNN = MaskNet->GetSegmentation(im,string(argv[4]),vstrImageFilenames[ni].replace(0,4,"")); //0 background y 1 foreground
            cv::Mat maskRCNNdil = maskRCNN.clone();
            cv::dilate(maskRCNN,maskRCNNdil, kernel);
            mask = mask - maskRCNNdil;
        }
        // Pass the image to the SLAM system
        SLAM.TrackMonoPainter(im, mask, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

		if (argc == 6)
        {
			//Here you can change the folder where the painted frame will store
            cv::imwrite(string(argv[5]) + "/PaintedFrames/" + vstrImageFilenames[ni],imgRGBOut[0]);	//It will be OK to comment this linge of code, where the painted frame will not be stored
			imgRGBOut.erase(imgRGBOut.begin());	//But comment this line might crush the System

        }

		
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);

    }

    // Stop all threads
    SLAM.Shutdown();
	
    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

