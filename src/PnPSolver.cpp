#include "PnPSolver.hpp"

PnPSolver::PnPSolver(){}

PnPSolver::~PnPSolver(){}

void PnPSolver::PnPSolve(const std::vector<cv::Point3f> &pt3f, 
            const std::vector<cv::Point2f> &pt2f, 
            const cv::Mat &cameraMatrix,
            cv::Mat &rvec,
            cv::Mat &tvec)
{
    cv::Mat inliers;
    cv::solvePnPRansac(pt3f, pt2f, cameraMatrix, cv::Mat(), rvec, tvec, 
            false, 100, 1.0, 100, inliers);    
}

void PnPSolver::PnPSolve(const std::vector<cv::Point2f> &ori_pt2f, 
            const std::vector<cv::Point2f> &des_pt2f, 
            const cv::Mat &cameraMatrix,
            cv::Mat &rvec,
            cv::Mat &tvec)
{
    std::vector<cv::Point3f> pt3f;
    for(int i = 0, _end = ori_pt2f.size(); i < _end; i++){
    }
}

void PnPSolver::FindEssentialMat(cv::Mat &fundamentalMat, cv::Mat &cameraMatrix, cv::Mat &result)
{
    cv::Mat cameraMatrixT = cameraMatrix.t();
    result = cameraMatrixT * fundamentalMat * cameraMatrix;
    std::cout << result << std::endl;
}

void PnPSolver::FindFundamentalMatRansac(std::vector<cv::Point2f> &pts1, 
        std::vector<cv::Point2f> &pts2,
        cv::Mat &result,
        cv::Mat &inliers)
{
    result = cv::findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 1, 0.999, inliers); 
    std::cout << result << std::endl;
}
