/**
* This file is a modified version of Dyna-SLAM.<https://github.com/BertaBescos/DynaSLAM>
*
* This file is part of DynaSLAM_MonoPainter.
* For more information see <https://github.com/Ocenal/DynaSLAM_MonoPainter>.
*
*/

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"
#include "MonoPainter.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>
#include<utility>



namespace ORB_SLAM2
{

class Tracking;
class Viewer;

class FrameDrawer
{
public:
    FrameDrawer(Map* pMap);

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);

    // Draw last processed frame.
    cv::Mat DrawFrame();

	cv::Mat GetProcessedMonoFrame();

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat mIm;
    int N;
    vector<cv::KeyPoint> mvCurrentKeys;
    vector<bool> mvbMap, mvbVO;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys;
    vector<int> mvIniMatches;
    int mState;

    Map* mpMap;

    std::mutex mMutex;

	MonoPainter::PainterObject* framepainter;
};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
