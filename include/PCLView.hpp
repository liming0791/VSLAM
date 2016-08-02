#include "stdio.h"

#include <iostream>

#include "opencv2/opencv.hpp"

#include <boost/thread/thread.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

class PCLView
{
public:
    PCLView();
    ~PCLView();
    void showPoints(cv::Mat P3Ds);
    void showRGBPoints(cv::Mat &frame, std::vector<cv::Point2f> &p2s, std::vector<cv::Point3f> &p3s);
};
