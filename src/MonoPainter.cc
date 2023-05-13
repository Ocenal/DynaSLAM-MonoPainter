/**
* This file is a modified version of Dyna-SLAM.<https://github.com/BertaBescos/DynaSLAM>
*
* This file is part of DynaSLAM_MonoPainter.
* For more information see <https://github.com/Ocenal/DynaSLAM_MonoPainter>.
*
*/

#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include "Frame.h"
#include "Map.h"
#include "MonoPainter.h"
#include "KeyFrame.h"

#include<mutex>
#include<utility>



namespace MonoPainter
{

static bool cmp(std::pair<float, ORB_SLAM2::KeyFrame*> &pair1, std::pair<float, ORB_SLAM2::KeyFrame*> &pair2)
{
	return pair1.first < pair2.first;
}

PainterObject::PainterObject(std::vector<cv::Mat>& initialization, ORB_SLAM2::Map* AnyMap, const string &strSettingsFile):
	imageOutPut(initialization), mpMap(AnyMap)
{
	std::cerr<<"\nLoading intrisic parameters of camera for MonoPainter..."<<std::endl;
	cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
	
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(CameraIntrinsic);
	std::cerr<<"\nIntrisic parameters of camera for MonoPainter Loaded successfully."<<std::endl;
	TestImageRGB = cv::Mat::zeros(480,640,CV_8U);
	Image2Display = make_pair(TestImageRGB, ShowedImageId);
	
}

void PainterObject::UpdatePainter(const ORB_SLAM2::Frame &frame)
{
	imRGB = frame.mImRGB.clone();
	mask = frame.mImMask.clone();	
}

void PainterObject::MaskTheImage()
{
	cv::Mat channels[3];
	cv::split(imRGB, channels);
	channels[0] = channels[0].mul(mask);
	channels[1] = channels[1].mul(mask);
	channels[2] = channels[2].mul(mask);
	
	std::vector<cv::Mat> arrayToMerge;
	cv::Mat maskedimage;
    arrayToMerge.push_back(channels[0]);
    arrayToMerge.push_back(channels[1]);
    arrayToMerge.push_back(channels[2]);
    cv::merge(arrayToMerge, maskedimage);
	
	
	TestImageRGB = maskedimage.clone();
	ShowedImageId++;
	Image2Display.second = ShowedImageId;
	imageOutPut.push_back(TestImageRGB);
}

std::pair<cv::Mat,int> PainterObject::GetProcessedFrame()
{
	return Image2Display;
}


//=============================================================================================================
//Below laid the code copy from DynaSlam/src/Geometry.cc
cv::Mat PainterObject::rotm2euler(const cv::Mat &R){
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6;
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    cv::Mat res = (cv::Mat_<double>(1,3) << x, y, z);
    return res;
}

bool PainterObject::isRotationMatrix(const cv::Mat &R){
    cv::Mat Rt;
    transpose(R,Rt);
    cv::Mat shouldBeIdentity = Rt*R;
    cv::Mat I = cv::Mat::eye(3,3,shouldBeIdentity.type());
    return norm(I,shouldBeIdentity) < 1e-6;
}

//=============================================================================================================



std::vector<std::pair<float,ORB_SLAM2::KeyFrame*>> PainterObject::GetSortedRefKeyFramePairs(const ORB_SLAM2::Frame &frame, const float &ratio)
{	
	std::vector<ORB_SLAM2::KeyFrame*> AvailableKeyFrames = mpMap->GetAllKeyFrames();
	cv::Mat currTcw = frame.mTcw.clone();
	if(currTcw.empty()){exit(-1);}
	cv::Mat rot1 = currTcw.rowRange(0,3).colRange(0,3);
    cv::Mat eul1 = rotm2euler(rot1);
	cv::Mat trans1 = currTcw.rowRange(0,3).col(3);
	
	std::vector<std::pair<float,ORB_SLAM2::KeyFrame*>> AssessedKeyFrames;
	int count = 0;
	for(std::vector<ORB_SLAM2::KeyFrame*>::iterator i = std::begin(AvailableKeyFrames),\
		iend = std::end(AvailableKeyFrames);\
			i != iend; i++)
	{
		ORB_SLAM2::KeyFrame* currKF = *i;
		cv::Mat kfTcw = currKF->GetPose();
		
		cv::Mat rot2 = kfTcw.rowRange(0,3).colRange(0,3);
        cv::Mat eul2 = rotm2euler(rot2);
        double distRot = cv::norm(eul2,eul1,cv::NORM_L2);
		
		cv::Mat trans2 = kfTcw.rowRange(0,3).col(3);
        double dist = cv::norm(trans2,trans1,cv::NORM_L2);

		double FinalDist = 0.7*distRot + 0.3*dist;
		
        std::pair<float,ORB_SLAM2::KeyFrame*> temp = make_pair(FinalDist, currKF);
		AssessedKeyFrames.push_back(temp);
		count++;
	}

	std::sort(AssessedKeyFrames.begin(), AssessedKeyFrames.end(), MonoPainter::cmp);
	return std::vector<std::pair<float,ORB_SLAM2::KeyFrame*>>({
			AssessedKeyFrames.begin(), AssessedKeyFrames.begin() + (int)(count*ratio)});
}

std::pair<std::vector<float>,std::vector<ORB_SLAM2::KeyFrame*>> PainterObject::GetSplitedPairs(const ORB_SLAM2::Frame &frame, const float &ratio)
{	std::vector<pair<float,ORB_SLAM2::KeyFrame*>> AssessedKeyFrames = \
		GetSortedRefKeyFramePairs(frame,ratio);
	std::vector<float> FinalDistance;
	std::vector<ORB_SLAM2::KeyFrame*> FinalKeyFrames;
	for(std::vector<std::pair<float,ORB_SLAM2::KeyFrame*>>::iterator i = AssessedKeyFrames.begin(),\
			iend = AssessedKeyFrames.end();\
				i != iend; i++)
	{
		const std::pair<float,ORB_SLAM2::KeyFrame*> TempValue = *i; 
		FinalDistance.push_back(TempValue.first);
		FinalKeyFrames.push_back(TempValue.second);
	}
	float MaxElement = *std::max_element(FinalDistance.begin(),FinalDistance.end());
	if(MaxElement > 0)
	{
		for(std::vector<float>::iterator i = FinalDistance.begin(),
				iend = FinalDistance.end();
					i != iend; i++)
		{
			*i = (*i)/MaxElement;
		}
	}
	
	return std::make_pair(FinalDistance,FinalKeyFrames);
}

void PainterObject::ProjectMapPoints(const ORB_SLAM2::Frame &frame, const float &ratio, const float &radius)
{
	DiffusionRadius = radius;
	std::vector<MonoPainter::PainterData> AssociatedData = GetAssociatedData(frame, ratio);
	
	//The logic of this segment of code were hard to figure out,
	//since it was revised during debuging through a gedit window
	//What might be helpful is that 
	//the vector "imageOutPut" is a reference vector for exporting painted frames
	bool PackageEmpty = AssociatedData.empty();
	if(PackageEmpty){
		imageOutPut.pop_back();
		imageOutPut.push_back(TestImageRGB);
		return;
	}else{
		bool PixelsEmpty = ((*AssociatedData.begin()).keypixels.empty());
		if(PixelsEmpty){
			imageOutPut.pop_back();
			imageOutPut.push_back(TestImageRGB);
			return;
		}
		//You can adjust the timestamp where the first frame to be painted begins
		//For example, if you have a sequence of 500 images, you can set "ShowedImageId > 200",
		//then the first painted frame will be the 200th image of your sequence.
		//Since it takes time for DynaSLAM to finish initialization in monocular state.		
		if(ShowedImageId > 0){
			Diffusion(AssociatedData);
			imageOutPut.pop_back();
			imageOutPut.push_back(TestImageRGB);
		}else{
			//"DrawMapPoints" can show how many map points were paintable in current frame
			DrawMapPoints(AssociatedData);
			imageOutPut.pop_back();
			imageOutPut.push_back(TestImageRGB);		
		}
	}
}
 
void PainterObject::Diffusion(std::vector<MonoPainter::PainterData> &Association)
{
	cv::Mat channel_00 = cv::Mat::zeros(imRGB.rows, imRGB.cols, CV_32F);
	cv::Mat channel_01 = cv::Mat::zeros(imRGB.rows, imRGB.cols, CV_32F);
	cv::Mat channel_02 = cv::Mat::zeros(imRGB.rows, imRGB.cols, CV_32F);
	cv::Mat AccumulationCount_00 = cv::Mat::ones(imRGB.rows, imRGB.cols, CV_32F);
	cv::Mat AccumulationCount_01 = cv::Mat::ones(imRGB.rows, imRGB.cols, CV_32F);
	cv::Mat AccumulationCount_02 = cv::Mat::ones(imRGB.rows, imRGB.cols, CV_32F);
	
	for(std::vector<MonoPainter::PainterData>::iterator i = Association.begin(),
			iend = Association.end();
				i != iend; i++)
	{
		MonoPainter::PainterData TempData = *i;
		cv::Mat KeyFrameImRGB = TempData.keyframe->mImRGB.clone();
		cv::Mat KeyFrameImMask = TempData.keyframe->mImMask.clone();
		std::vector<pair<int,int>> KeyPixels = TempData.keypixels;
		std::vector<pair<int,int>> CurrPixels = TempData.currpixels;
		cv::Mat channels[3];
		cv::split(KeyFrameImRGB, channels);
		ChannelAccumulator(channel_00, AccumulationCount_00, channels[0], KeyFrameImMask, KeyPixels, CurrPixels);
		ChannelAccumulator(channel_01, AccumulationCount_01, channels[1], KeyFrameImMask, KeyPixels, CurrPixels);
		ChannelAccumulator(channel_02, AccumulationCount_02, channels[2], KeyFrameImMask, KeyPixels, CurrPixels);
	}
	cv::Mat newMask = mask.clone();
	cv::Mat tempMask = cv::Mat::ones(mask.size(),mask.type());
	newMask = tempMask - mask;

	channel_00 = channel_00.mul(1/AccumulationCount_00);
	channel_00.convertTo(channel_00, mask.type());
	channel_00 = channel_00.mul(newMask);
	channel_00.convertTo(channel_00, CV_8U);
	
	
	channel_01 = channel_01.mul(1/AccumulationCount_01);
	channel_01.convertTo(channel_01, mask.type());
	channel_01 = channel_01.mul(newMask);
	channel_01.convertTo(channel_01, CV_8U);
	
	channel_02 = channel_02.mul(1/AccumulationCount_02);
	channel_02.convertTo(channel_02, mask.type());
	channel_02 = channel_02.mul(newMask);
	channel_02.convertTo(channel_02, CV_8U);
	
	std::vector<cv::Mat> ArrayToMerge;
	cv::Mat DiffusedImage;
    ArrayToMerge.push_back(channel_00);
    ArrayToMerge.push_back(channel_01);
    ArrayToMerge.push_back(channel_02);
    cv::merge(ArrayToMerge, DiffusedImage);
	TestImageRGB = TestImageRGB + DiffusedImage;
	Image2Display = make_pair(TestImageRGB, ShowedImageId);
	
}

void PainterObject::ChannelAccumulator(cv::Mat &Channel, cv::Mat &AccumulationCount, cv::Mat &KeyChannel, cv::Mat &KeyMask, std::vector<pair<int,int>> &KeyPixels, std::vector<pair<int,int>> &CurrPixels)
{
	std::vector<pair<int,int>>::iterator i = KeyPixels.begin();
	std::vector<pair<int,int>>::iterator iend = KeyPixels.end();
	std::vector<pair<int,int>>::iterator j = CurrPixels.begin();
	std::vector<pair<int,int>>::iterator jend = CurrPixels.end();
	int half_n = std::ceil(DiffusionRadius);
	int RowConstrain = imRGB.rows;
	int ColConstrain = imRGB.cols;
	for(; i != iend && j != jend; i++, j++)
	{
		std::pair<int,int> KeyPixel = *i;
		std::pair<int,int> CurrPixel = *j;
		//Diffusing around the matched pixel in a rounded region
		for(int row_inc = -half_n; row_inc <= half_n; row_inc++)
		{
			for(int col_inc = -half_n; col_inc <= half_n; col_inc++)
			{
				int KeyNearDim00 = KeyPixel.first + row_inc;
				int KeyNearDim01 = KeyPixel.second + col_inc;
				int CurrNearDim00 = CurrPixel.first + row_inc;
				int CurrNearDim01 = CurrPixel.second + col_inc;
				bool NotCenter = !((row_inc == 0) && (col_inc == 0));
				bool InKeyFrame = (KeyNearDim00 > 0 && KeyNearDim00 < RowConstrain && KeyNearDim01 > 0 && KeyNearDim01 < ColConstrain);
				bool InCurrFrame = (CurrNearDim00 > 0 && CurrNearDim00 < RowConstrain && CurrNearDim01 > 0 && CurrNearDim01 < ColConstrain);
				bool NotMasked = (KeyMask.at<uchar>(KeyNearDim00, KeyNearDim01) != 0);
				bool RadiusConstrain = ((row_inc*row_inc + col_inc*col_inc) < (DiffusionRadius*DiffusionRadius));
				if(NotCenter && InKeyFrame && InCurrFrame && NotMasked && RadiusConstrain)
				{	float weight = GetWeight(KeyChannel, KeyMask,KeyPixel);
					AccumulationCount.at<float>(CurrNearDim00, CurrNearDim01) += weight;
					Channel.at<float>(CurrNearDim00, CurrNearDim01) += \
						weight*(float)KeyChannel.at<uchar>(KeyNearDim00, KeyNearDim01);
				}else{
					continue;
				}
			}
		}
	}
}

float PainterObject::GetWeight(cv::Mat &KeyChannel, cv::Mat &KeyMask, std::pair<int,int> &KeyPixel)
{
	int Dim_00 = KeyPixel.first;
	int Dim_01 = KeyPixel.second;
	int half_n = std::ceil(WeightRadius);
	float WeightAccumulator = 0;
	float WeightCount = 0;
	for(int row_inc = -half_n; row_inc <= half_n; row_inc++)
	{
		for(int col_inc = -half_n; col_inc <= half_n; col_inc++)
		{
			int TempDim_00 = Dim_00 + row_inc;
			int TempDim_01 = Dim_01 + col_inc;
			bool NotCenter = !(row_inc == 0 && col_inc == 0);
			bool InKeyFrame = (TempDim_00 > 0 && TempDim_00 < (imRGB.rows - 1) && TempDim_01 > 0 && TempDim_01 < (imRGB.cols - 1));
			bool NotMasked = (KeyMask.at<uchar>(TempDim_00, TempDim_01) != 0);
			if(NotCenter && NotMasked &&InKeyFrame)
			{
				WeightAccumulator += KeyChannel.at<uchar>(TempDim_00, TempDim_01);
				WeightCount++;
			}else{
				continue;
			}
		}
	}
	WeightCount = WeightCount*10;
	return (WeightAccumulator/WeightCount);
}

void PainterObject::DrawMapPoints(std::vector<MonoPainter::PainterData> &Association)
{
	
	for(std::vector<MonoPainter::PainterData>::iterator i = Association.begin(),
			iend = Association.end();
				i != iend; i++)
	{
		const MonoPainter::PainterData &Data = *i;
		const float &Dist = Data.distance;
		std::vector<std::pair<int,int>> CurrPixels = Data.currpixels;
		const float radius = 5;
		int r,g,b;
		for(std::vector<std::pair<int,int>>::iterator j = CurrPixels.begin(),
				jend = CurrPixels.end();
					j != jend; j++)
		{
			std::pair<int,int> Pixel = *j;
			cv::Point2f pt1,pt2,pt3;
        	pt1.y = Pixel.first - radius;	//dim 00 set to y
        	pt1.x = Pixel.second - radius;	//dim 01 set to x
        	pt2.y = Pixel.first + radius;
        	pt2.x = Pixel.second + radius;
			pt3.y = Pixel.first;
			pt3.x = Pixel.second;
			r = (int)255*(1-Dist);
			b = (int)255*Dist;
			g = (int)255*(0.1);
			cv::rectangle(TestImageRGB,pt1,pt2,cv::Scalar(r,g,b));
            cv::circle(TestImageRGB,pt3,2,cv::Scalar(r,g,b),-1);
		}
	}
				
}


//PixelPair<dim 00, dim 01>	--<row_index, col_index>
std::vector<MonoPainter::PainterData> PainterObject::GetAssociatedData(const ORB_SLAM2::Frame &frame, const float &ratio)
{	std::pair<std::vector<float>,std::vector<ORB_SLAM2::KeyFrame*>> Pair_DistVec_KFVec = GetSplitedPairs(frame, ratio);
	std::vector<float> &DistVec = Pair_DistVec_KFVec.first;
	std::vector<ORB_SLAM2::KeyFrame*> &KFVec = Pair_DistVec_KFVec.second;
	
	cv::Mat KO;
	cv::Mat currTcw = frame.mTcw.clone();
	cv::hconcat(CameraIntrinsic,cv::Mat::zeros(3,1,CameraIntrinsic.type()), KO);
	std::vector<MonoPainter::PainterData> FinalVector;
	std::vector<float>::iterator k = DistVec.begin();
	std::vector<float>::iterator kend = DistVec.end();
	std::vector<ORB_SLAM2::KeyFrame*>::iterator i = KFVec.begin();
	std::vector<ORB_SLAM2::KeyFrame*>::iterator iend = KFVec.end();
	for(;i != iend && k != kend;i++, k++)
	{
		float Dist = *k;
		ORB_SLAM2::KeyFrame* KeyFrame = *i;
		std::vector<std::pair<int,int>> CurrPixels;
		std::vector<std::pair<int,int>> KeyPixels;
		
		MonoPainter::PainterData TempData;	

		cv::Mat keyTcw = KeyFrame->GetPose();
		
		std::set<ORB_SLAM2::MapPoint*> MapPoints= KeyFrame->GetMapPoints();
		for(std::set<ORB_SLAM2::MapPoint*>::iterator j = MapPoints.begin(),\
				jend = MapPoints.end();\
					j != jend; j++)
		{
			ORB_SLAM2::MapPoint* MapPoint = *j;
			cv::Mat WorldPos = MapPoint->GetWorldPos();
			cv::Mat HomoWorldPos; 
			cv::vconcat(WorldPos,cv::Mat::eye(1,1,WorldPos.type()),HomoWorldPos);
			
			cv::Mat HomoKeyPos = keyTcw*HomoWorldPos;
			float Zk = HomoKeyPos.at<float>(2);
			
			cv::Mat HomoCurrPos = currTcw*HomoWorldPos;
			float Zc = HomoCurrPos.at<float>(2);
			if(Zc > 0 && Zk > 0){
				cv::Mat KeyPixel = KO*HomoKeyPos/Zk;
				int keyu = (int)KeyPixel.at<float>(0);	//for dim 01 of image, which is cols
				int keyv = (int)KeyPixel.at<float>(1);	//for dim 00 of image, which is rows
				
				cv::Mat CurrPixel = KO*HomoCurrPos/Zc;
				int curru = (int)CurrPixel.at<float>(0);	//for dim 01 of image, which is cols
				int currv = (int)CurrPixel.at<float>(1);	//for dim 00 of image, which is rows
				

				bool keyinFrame = (keyv > 1 && keyv < (imRGB.rows - 1) && keyu > 1 && keyu < (imRGB.cols - 1));
				bool currinFrame = (currv > 1 && currv < (imRGB.rows - 1) && curru > 1 && curru < (imRGB.cols - 1));
				if(keyinFrame && currinFrame && mask.at<uchar>(currv,curru) == 0){
					std::pair<int,int> TempKeyPixel(keyv,keyu);
					KeyPixels.push_back(TempKeyPixel);		//Now we have recorded the pigment for painting 
				
					std::pair<int,int> TempCurrPixel(currv,curru);
					CurrPixels.push_back(TempCurrPixel);	//Now we have recorded the coordinates for painting
				}else{
					continue;	//Discard it
				}
			}else{
				continue;	//Discard it
			}
		}//All MapPoints of a KeyFrame have been processed
		TempData.distance = Dist;
		TempData.keyframe = KeyFrame;
		TempData.keypixels = KeyPixels;
		TempData.currpixels = CurrPixels;
		FinalVector.push_back(TempData);

		KeyPixels.clear();
		CurrPixels.clear();	
	}//All KeyFrames have been processed
	return FinalVector;
}


}



