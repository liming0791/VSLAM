#include "stdio.h"
#include "time.h"

#include <iostream>
#include <vector>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"

class PnPSolver
{
public:
    PnPSolver();
    ~PnPSolver();
    void PnPSolve(const std::vector<cv::Point3f> &pt3f, 
            const std::vector<cv::Point2f> &pt2f, 
            const cv::Mat &cameraMatrix,
            cv::Mat &rvec,
            cv::Mat &tvec);
    void PnPSolve(const std::vector<cv::Point2f> &ori_pt2f, 
            const std::vector<cv::Point2f> &des_pt2f, 
            const cv::Mat &cameraMatrix,
            cv::Mat &rvec,
            cv::Mat &tvec);
    void FindFundamentalMatRansac(std::vector<cv::Point2f> &pts1, 
            std::vector<cv::Point2f> &pts2, 
            cv::Mat &result,
            cv::Mat &inliers);
    void FindEssentialMat(cv::Mat &fundamentalMat, cv::Mat &cameraMatrix, cv::Mat &result);

};
