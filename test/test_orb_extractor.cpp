#include "stdio.h"
#include "time.h"

#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "PnPSolver.hpp"

void drawLink(cv::Mat& img, cv::Point& p1, cv::Point& p2, cv::Scalar& color)
{
	cv::circle(img, p1, 3, color, 1, 8, 0);
	cv::circle(img, p2, 3, color, 1, 8, 0);
	cv::line(img, p1, p2, color);
}

void drawMatchResult(std::string win_name , cv::Mat& src, cv::Mat& des, 
		std::vector<cv::KeyPoint*>& srcKeyPoints, std::vector<cv::KeyPoint*>& desKeyPoints,
        cv::Mat mask = cv::Mat())
{

	int newRows = src.rows > des.rows ? src.rows : des.rows;
	int newCols = src.cols + des.cols;

	cv::Mat resultMat(newRows, newCols, CV_8UC3);

	src.copyTo(resultMat(cv::Rect(0, 0, src.cols, src.rows)));
	des.copyTo(resultMat(cv::Rect(src.cols, 0, des.cols, des.rows)));

	cv::RNG rng(12345);
    int num_match = 0;
	for(size_t t = 0; t < srcKeyPoints.size(); t++){
        if(!mask.empty() && !mask.data[t]) continue;
		cv::KeyPoint* kpt1 = srcKeyPoints[t];
		cv::KeyPoint* kpt2 = desKeyPoints[t];
		cv::Point pt1(kpt1->pt.x, kpt1->pt.y);
		cv::Point pt2(kpt2->pt.x + src.cols, kpt2->pt.y);
		cv::Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawLink(resultMat, pt1, pt2, color);
        num_match++;
	}
    printf("num match: %d\n", num_match);

	cv::namedWindow(win_name);
	cv::imshow(win_name ,resultMat);

	while(cv::waitKey()<0){}

}

void getEssentialMat(cv::Mat &frame1, cv::Mat &frame2,std::vector<cv::KeyPoint*> &out1, std::vector<cv::KeyPoint*> &out2, cv::Mat &result)
{
    //calculate fundamentalmat and essentialmat
    PnPSolver solver;
    std::vector<cv::Point2f> ori_pt;
    std::vector<cv::Point2f> des_pt;
    cv::Mat fundamentalmat;
    cv::Mat inliers;
    double fx= 616.662852;
    double fy= 618.719985;
    double cx= 323.667190;
    double cy= 236.399432;

    double cameraArray[9] = 
    {
        fx, 0 , cx,
        0,  fy, cy,
        0,  0,  1
    };
    cv::Mat cameraMatrix(3, 3, CV_64FC1, cameraArray);
    
    for(int i = 0, _end = out1.size(); i<_end; i++){
        ori_pt.push_back(out1[i]->pt);
        des_pt.push_back(out2[i]->pt);
    }
    printf("fundamentalmat:\n");
    solver.FindFundamentalMatRansac(ori_pt, des_pt, fundamentalmat, inliers);
    printf("essentialmat:\n");
    solver.FindEssentialMat(fundamentalmat, cameraMatrix, result);

    //show inliers
    drawMatchResult("inliers", frame1, frame2,out1, out2, inliers);
   
}

void getRt(cv::Mat &essentialmat, std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts)
{
    cv::Mat u, vt, w;
    double W[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    cv::Mat WMat(3, 3, CV_64FC1, W);
    cv::SVD::compute(essentialmat, w, u, vt);

    Rs.push_back(u * WMat * vt);
    Rs.push_back(u * WMat.t() * vt.t());

    ts.push_back(u.col(2));
    ts.push_back(-u.col(2));


    std::cout << "R1: " << std::endl << Rs[0] << std::endl;
    std::cout << "R2: " << std::endl << Rs[1] << std::endl;
    std::cout << "t1: " << std::endl << ts[0] << std::endl;
    std::cout << "t1: " << std::endl << ts[1] << std::endl;
    
}

int main(int argc, char** argv )
{
    if(argc < 3){
        printf("need more argument\n");
        return 0;
    }

    cv::Mat frame1, frame2, gray1, gray2;

    frame1 = cv::imread(argv[1]);
    frame2 = cv::imread(argv[2]);

    float re_w = 640.f;

    cv::resize(frame1, frame1, cv::Size(re_w, frame1.rows * re_w/ frame1.cols ));
    cv::resize(frame2, frame2, cv::Size(re_w, frame2.rows * re_w/ frame2.cols ));

    cv::cvtColor(frame1, gray1, CV_BGR2GRAY);
    cv::cvtColor(frame2, gray2, CV_BGR2GRAY);

    //orb feature extract
    ORBextractor orb_extractor(1000, 1.2, 8, 20, 7);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat dc1, dc2;
    clock_t time = clock();
    orb_extractor(gray1, cv::Mat(), kp1, dc1);
    orb_extractor(gray2, cv::Mat(), kp2, dc2); 
    printf("time extract feature: %f\n", (clock()-time)/(double)CLOCKS_PER_SEC);

    //orb feature match
    std::vector<cv::KeyPoint*> out1, out2;
    ORBmatcher orb_matcher(0.9, true);
    time = clock();
    orb_matcher.SearchByKeypointsInGrid(
            frame1.cols, 
            frame1.rows, 
            kp1, dc1, kp2, dc2, 
            out1, out2);
    printf("time match: %f\n", (clock()-time)/(double)CLOCKS_PER_SEC);

    //draw orb result
    drawMatchResult("orb result", frame1, frame2, out1, out2);

    //get essentialmat
    cv::Mat essentialmat;
    getEssentialMat(frame1, frame2, out1, out2, essentialmat);

    //SVD the essentialmat
    std::vector<cv::Mat> Rs, ts;
    getRt(essentialmat, Rs, ts); 

    //opencv feature extract
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;
    cv::initModule_nonfree();
    _detector = cv::FeatureDetector::create("GridSIFT");
    _descriptor = cv::DescriptorExtractor::create("SIFT");

    std::vector<cv::KeyPoint> ocv_kp1, ocv_kp2;
    cv::Mat desp1, desp2;

    time = clock();
    _detector->detect(frame1, ocv_kp1);
    _detector->detect(frame2, ocv_kp2);
    _descriptor->compute(frame1, ocv_kp1, desp1);
    _descriptor->compute(frame2, ocv_kp2, desp2);
    printf("time extract feature opencv: %f\n", (clock()-time)/(double)CLOCKS_PER_SEC);

    //opencv feature match
    std::vector<cv::DMatch> matches;
    cv::FlannBasedMatcher matcher;
    time = clock();
    matcher.match(desp1, desp2, matches);

    //filter good matched
    std::vector< cv::DMatch > goodMatches;
    double minDis = 9999, maxDis = -1;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
        if(matches[i].distance > maxDis)
            maxDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 0.25*(maxDis - minDis) + minDis)
            goodMatches.push_back( matches[i] );
    } 
    printf("time match feature opencv: %f\n", (clock()-time)/(double)CLOCKS_PER_SEC);
    
    //draw opencv result
    cv::Mat imageMatches;
    cv::drawMatches(frame1, ocv_kp1, frame2, ocv_kp2, goodMatches, imageMatches);
    cv::namedWindow("opencv result");
    cv::imshow("opencv result", imageMatches);
    while(cv::waitKey()<0){}

    //get essentialmat
    std::vector<cv::KeyPoint*> ocv_out1;
    std::vector<cv::KeyPoint*> ocv_out2;

    for(int i = 0, _end = goodMatches.size(); i<_end; i++ ){
        ocv_out1.push_back(&ocv_kp1[goodMatches[i].queryIdx]);
        ocv_out2.push_back(&ocv_kp2[goodMatches[i].trainIdx]);
    }

    cv::Mat ocv_essentialmat;
    getEssentialMat(frame1, frame2, ocv_out1, ocv_out2, ocv_essentialmat);

    //SVD essentialmat
    std::vector<cv::Mat> ocv_Rs, ocv_ts;
    getRt(ocv_essentialmat, ocv_Rs, ocv_ts);
    

    return 0;
}