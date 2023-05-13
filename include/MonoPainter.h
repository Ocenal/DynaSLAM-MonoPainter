/**
* This file is a modified version of Dyna-SLAM.<https://github.com/BertaBescos/DynaSLAM>
*
* This file is part of DynaSLAM_MonoPainter.
* For more information see <https://github.com/Ocenal/DynaSLAM_MonoPainter>.
*
*/

#ifndef MONOPAINTER_H
#define MONOPAINTER_H

#include <string>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include "Frame.h"

#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Geometry.h"


#include<mutex>
#include<utility>


namespace MonoPainter
{

static bool cmp(std::pair<float,ORB_SLAM2::KeyFrame*> &pair1, std::pair<float,ORB_SLAM2::KeyFrame*> &pair2);

typedef struct DataExtraction
{
	float distance;
	ORB_SLAM2::KeyFrame* keyframe;
	//PixelPair<dim 00, dim 01>	--<row_index, col_index>
	std::vector<std::pair<int,int>> keypixels;
	std::vector<std::pair<int,int>> currpixels;
}PainterData;

class PainterObject
{
private:
	//raw material
	cv::Mat imRGB;
	cv::Mat mask;
	ORB_SLAM2::Map* mpMap;
	cv::Mat CameraIntrinsic;
	float DiffusionRadius = 10;
	float WeightRadius = 5;

	//global communication
	std::vector<cv::Mat>& imageOutPut;

	//image to be viewed
	std::pair<cv::Mat,int> Image2Display; 
	cv::Mat TestImageRGB;
	int ShowedImageId = 0;

public:
	PainterObject();
	PainterObject(std::vector<cv::Mat>& initialization, ORB_SLAM2::Map* AnyMap, const string &strSettingsFile);
	~PainterObject() = default;
	
	void MaskTheImage();
	
	cv::Mat rotm2euler(const cv::Mat &R);
	bool isRotationMatrix(const cv::Mat &R);
	
	void ProjectMapPoints(const ORB_SLAM2::Frame &frame, const float &ratio = 1., const float &radius = 10);
	std::vector<std::pair<float,ORB_SLAM2::KeyFrame*>> GetSortedRefKeyFramePairs(const ORB_SLAM2::Frame &frame, const float &ratio = 1.);
	std::pair<std::vector<float>,std::vector<ORB_SLAM2::KeyFrame*>> GetSplitedPairs(const ORB_SLAM2::Frame &frame, const float &ratio = 1.);
	std::vector<MonoPainter::PainterData> GetAssociatedData(const ORB_SLAM2::Frame &frame, const float &ratio = 1.);
	void DrawMapPoints(std::vector<MonoPainter::PainterData> &Association);
	void Diffusion(std::vector<MonoPainter::PainterData> &Association);
	//OutPut OutPutHelp material_00 material_01 material_02 material_03 
	void ChannelAccumulator(cv::Mat &Channel, cv::Mat &AccumulationCount, cv::Mat &KeyChannel, cv::Mat &KeyMask, std::vector<pair<int,int>> &KeyPixels, std::vector<pair<int,int>> &CurrPixels);
	float GetWeight(cv::Mat &KeyChannel, cv::Mat &KeyMask, std::pair<int,int> &KeyPixel);
	
	void UpdatePainter(const ORB_SLAM2::Frame &frame);
	std::pair<cv::Mat,int> GetProcessedFrame();
};

}


#endif	//MONOPAINTER_H

