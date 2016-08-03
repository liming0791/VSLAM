#include "stdio.h"
#include "time.h"

#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "Eigen/Dense"

#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "PnPSolver.hpp"
#include "PCLView.hpp"

using namespace std;
using namespace cv;

std::string getImageType(cv::Mat &mat)
{
    int number = mat.type();
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

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

void getEssentialMat(cv::Mat &frame1, cv::Mat &frame2,std::vector<cv::KeyPoint*> &out1, std::vector<cv::KeyPoint*> &out2, cv::Mat &result, cv::Mat &inliers)
{
    //calculate fundamentalmat and essentialmat
    PnPSolver solver;
    std::vector<cv::Point2f> ori_pt;
    std::vector<cv::Point2f> des_pt;
    cv::Mat fundamentalmat;
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
    std::cout << getImageType(fundamentalmat) << std::endl;
    printf("essentialmat:\n");
    solver.FindEssentialMat(fundamentalmat, cameraMatrix, result);
    std::cout << getImageType(result) << std::endl;

    //show inliers
    drawMatchResult("inliers", frame1, frame2,out1, out2, inliers);
   
}

void getRt(cv::Mat &essentialmat, std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts)
{
    cv::Mat u, vt, w;
    double W[9] = 
    {
        0, -1, 0, 
        1, 0, 0, 
        0, 0, 1
    };
    cv::Mat WMat(3, 3, CV_64FC1, W);
    cv::SVD::compute(essentialmat, w, u, vt);

    Rs.push_back(u * WMat * vt);
    Rs.push_back(u * WMat.t() * vt);

    ts.push_back(u.col(2));
    ts.push_back(-u.col(2));


    std::cout << "R1: " << std::endl << Rs[0] << std::endl;
    std::cout << "R2: " << std::endl << Rs[1] << std::endl;
    std::cout << "t1: " << std::endl << ts[0] << std::endl;
    std::cout << "t1: " << std::endl << ts[1] << std::endl;
    
}



void getPointsFromHomogeneous(cv::Mat &h, std::vector<cv::Point3f> &pts){

    std::cout << h << std::endl << std::endl;

    float* ptr0 = h.ptr<float>(0);    
    float* ptr1 = h.ptr<float>(1);    
    float* ptr2 = h.ptr<float>(2);    
    float* ptr3 = h.ptr<float>(3);    

    pts.clear();

    float z = 0;
    for(int i = 0; i< h.cols;i++){
        z = ptr3[i];
        pts.push_back(cv::Point3f(ptr0[i]/z, ptr1[i]/z, ptr2[i]/z));
    }
}

bool Point3fRight(std::vector<cv::Point3f> &pt3f, cv::Mat &projMat){
    for(int i = 0, _end = pt3f.size(); i < _end; i++){
        if(pt3f[i].z<0 || (projMat.at<float>(8)*pt3f[i].x + projMat.at<float>(9)*pt3f[i].y + projMat.at<float>(10)*pt3f[i].z + projMat.at<float>(11))<0){
            return false;
        }
    }
    return true;
}

Mat_<float> LinearLSTriangulation(Point3f u,       //homogenous image point (u,v,1)
        Matx34f P,       //camera 1 matrix
        Point3f u1,      //homogenous image point in 2nd camera
        Matx34f P1       //camera 2 matrix
        )
{
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    

    //solve using eigen
    //Eigen::MatrixXf A(4,3);
    //A << u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
    //     u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
    //     u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
    //     u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2);
    //Eigen::Vector4f B;
    //B << -(u.x*P(2, 3) - P(0, 3)),
    //     -(u.y*P(2, 3) - P(1, 3)),
    //     -(u1.x*P1(2, 3) - P1(0, 3)),
    //     -(u1.y*P1(2, 3) - P1(1, 3));
    //Eigen::Vector3f x = A.colPivHouseholderQr().solve(B);
    //Mat_<float> X(3,1, CV_32FC1);
    //X.at<float>(0) = x[0];
    //X.at<float>(1) = x[1];
    //X.at<float>(2) = x[2];
    //solve using eigen done
    //
    //solve using opencv
    Matx43f A((u.x*P(2, 0) - P(0, 0)) , (u.x*P(2, 1) - P(0, 1)) , (u.x*P(2, 2) - P(0, 2)) ,
            (u.y*P(2, 0) - P(1, 0)) , (u.y*P(2, 1) - P(1, 1)) , (u.y*P(2, 2) - P(1, 2)) ,
            (u1.x*P1(2, 0) - P1(0, 0)) , (u1.x*P1(2, 1) - P1(0, 1)), (u1.x*P1(2, 2) - P1(0, 2)) ,
            (u1.y*P1(2, 0) - P1(1, 0)) , (u1.y*P1(2, 1) - P1(1, 1)) , (u1.y*P1(2, 2) - P1(1, 2))  
            );
    Mat_<float> B = (Mat_<float>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) ,
            -(u.y*P(2, 3) - P(1, 3)) ,
            -(u1.x*P1(2, 3) - P1(0, 3)) ,
            -(u1.y*P1(2, 3) - P1(1, 3)) 
            );
    cv::Mat X;
    solve(A, B, X, DECOMP_SVD);
    //solve using opencv done
    
    cout << "u" << endl 
        << u << endl 
        << "u1" << endl 
        << u1 << endl 
        << "P" << endl 
        << P << endl 
        << "P1" << endl
        << P1 << endl 
        << "A" << endl 
        << A << endl 
        << "B" << endl
        << B << endl 
        << "X" << endl
        << X << endl << endl;

    return X;
}

Mat_<float> IterativeLinearLSTriangulation(Point3f u,    //homogenous image point (u,v,1)
        Matx34f P,          //camera 1 matrix
        Point3f u1,         //homogenous image point in 2nd camera
        Matx34f P1          //camera 2 matrix
        ) {

    double  EPSILON = 0.000001;

    double wi = 1, wi1 = 1;
    Mat_<float> X(4, 1);

    for (int i = 0; i < 10; i++) { //Hartley suggests 10 iterations at most
        Mat_<float> X_ = LinearLSTriangulation(u, P, u1, P1);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
        //recalculate weights
        float p2x = Mat_<float>(Mat_<float>(P).row(2)*X)(0);
        float p2x1 = Mat_<float>(Mat_<float>(P1).row(2)*X)(0);

        //breaking point
        if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        Matx43f A((u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
                (u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
                (u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
                (u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
                );
        Mat_<float> B = (Mat_<float>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) / wi,
                -(u.y*P(2, 3) - P(1, 3)) / wi,
                -(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
                -(u1.y*P1(2, 3) - P1(1, 3)) / wi1
                );

        solve(A, B, X_, DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    }

    return X;
}

//Triagulate points
void TriangulatePoints(const vector<Point2f>& pt_set1,
                       const vector<Point2f>& pt_set2,
                       const Matx33f& K,
                       const Matx34f& P,
                       const Matx34f& P1,
                       vector<Point3f>& pointcloud)
{
    pointcloud.clear();
    cout << "Triangulating...";
    double t = getTickCount();
    unsigned int pts_size = pt_set1.size();
    for (unsigned int i=0; i < pts_size; i++){
        Point2f kp = pt_set1[i];
        Point3f u(kp.x,kp.y,1.0);
        Point2f kp1 = pt_set2[i];
        Point3f u1(kp1.x,kp1.y,1.0);
        Mat X = IterativeLinearLSTriangulation(u,K*P,u1,K*P1);
        pointcloud.push_back(Point3f(X.at<float>(0),X.at<float>(1),X.at<float>(2)));
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Done."<< endl;
}

void triangulate(std::vector<cv::Point2f> &f1_pt, std::vector<cv::Point2f> &f2_pt, std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts, std::vector<cv::Point3f> &result){
    cv::Mat p4s1, p4s2, p4s3, p4s4;
    std::vector<cv::Point3f> pt_3ds1, pt_3ds2, pt_3ds3, pt_3ds4;

    float fx= 616.662852;
    float fy= 618.719985;
    float cx= 323.667190;
    float cy= 236.399432;

    float cameraArray[9] = 
    {
        fx, 0 , cx,
        0,  fy, cy,
        0,  0,  1
    };
    cv::Mat cameraMatrix(3, 3, CV_32FC1, cameraArray);

    float projArray1[12] = 
    {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    }; 
    cv::Mat projMat1(3, 4, CV_32FC1, projArray1);
    //projMat1 = cameraMatrix * projMat1;
    std::cout << projMat1 << std::endl <<std::endl;
    //cv::Mat cameraMatrixInv = cameraMatrix.inv(DECOMP_SVD);

    cv::Mat projMat2_1(3,4,CV_32FC1), 
        projMat2_2(3,4,CV_32FC1), 
        projMat2_3(3,4,CV_32FC1), 
        projMat2_4(3,4,CV_32FC1);

    std::vector< cv::Mat > projMat2_;
    projMat2_.push_back(projMat2_1);
    projMat2_.push_back(projMat2_2);
    projMat2_.push_back(projMat2_3);
    projMat2_.push_back(projMat2_4);

    Rs[0].copyTo(projMat2_[0](cv::Rect(0,0,3,3)));
    ts[0].copyTo(projMat2_[0](cv::Rect(3,0,1,3)));
    std::cout << projMat2_[0] << std::endl << std::endl;
    //projMat2_1 = cameraMatrix * projMat2_1;
    //std::cout << projMat2_1 << std::endl << std::endl;

    Rs[0].copyTo(projMat2_[1](cv::Rect(0,0,3,3)));
    ts[1].copyTo(projMat2_[1](cv::Rect(3,0,1,3)));
    std::cout << projMat2_[1] << std::endl << std::endl;
    //projMat2_2 = cameraMatrix * projMat2_2;
    //std::cout << projMat2_2 << std::endl << std::endl;

    Rs[1].copyTo(projMat2_[2](cv::Rect(0,0,3,3)));
    ts[0].copyTo(projMat2_[2](cv::Rect(3,0,1,3)));
    std::cout << projMat2_[2] << std::endl << std::endl;
    //projMat2_3 = cameraMatrix * projMat2_3;
    //std::cout << projMat2_3 << std::endl << std::endl;

    Rs[1].copyTo(projMat2_[3](cv::Rect(0,0,3,3)));
    ts[1].copyTo(projMat2_[3](cv::Rect(3,0,1,3)));
    std::cout << projMat2_[3] << std::endl << std::endl;
    //projMat2_4 = cameraMatrix * projMat2_4;
    //std::cout << projMat2_4 << std::endl << std::endl;

    cv::triangulatePoints(cameraMatrix*projMat1, cameraMatrix*projMat2_[0], f1_pt, f2_pt, p4s1);
    cv::triangulatePoints(cameraMatrix*projMat1, cameraMatrix*projMat2_[1], f1_pt, f2_pt, p4s2);
    cv::triangulatePoints(cameraMatrix*projMat1, cameraMatrix*projMat2_[2], f1_pt, f2_pt, p4s3);
    cv::triangulatePoints(cameraMatrix*projMat1, cameraMatrix*projMat2_[3], f1_pt, f2_pt, p4s4);

    getPointsFromHomogeneous(p4s1, pt_3ds1);
    getPointsFromHomogeneous(p4s2, pt_3ds2);
    getPointsFromHomogeneous(p4s3, pt_3ds3);
    getPointsFromHomogeneous(p4s4, pt_3ds4);
    
    //TriangulatePoints(f1_pt, f2_pt, cameraMatrix, projMat1, projMat2_[0], pt_3ds1);
    //TriangulatePoints(f1_pt, f2_pt, cameraMatrix, projMat1, projMat2_[1], pt_3ds2);
    //TriangulatePoints(f1_pt, f2_pt, cameraMatrix, projMat1, projMat2_[2], pt_3ds3);
    //TriangulatePoints(f1_pt, f2_pt, cameraMatrix, projMat1, projMat2_[3], pt_3ds4);


    std::vector< std::vector<cv::Point3f> > results;
    results.push_back(pt_3ds1);
    results.push_back(pt_3ds2);
    results.push_back(pt_3ds3);
    results.push_back(pt_3ds4);

    for(int i = 0; i < 4; i++){
        if(Point3fRight(results[i], projMat2_[i])){
            printf("pts_3d %d valid\n", i);
            result = results[i];
        }
        else
            printf("pts_3d %d invalid\n", i);

        cout << results[i] << endl;
    }

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
    cv::Mat inliers;
    getEssentialMat(frame1, frame2, out1, out2, essentialmat,inliers);

    //SVD the essentialmat
    std::vector<cv::Mat> Rs, ts;
    getRt(essentialmat, Rs, ts); 


    std::vector<cv::Point2f> f1_pt, f2_pt;
    for(int i= 0 ,_end = inliers.rows; i < _end; i++){
        if(inliers.data[i]){
            f1_pt.push_back(out1[i]->pt);
            f2_pt.push_back(out2[i]->pt);
        }
    }

    //triangulate
    std::vector<cv::Point3f> pts_3d;
    triangulate(f1_pt, f2_pt, Rs, ts, pts_3d);
    
    
    
    PCLView view;
    view.showRGBPoints(frame1, f1_pt, pts_3d);

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
    cv::Mat ocv_inliers;
    getEssentialMat(frame1, frame2, ocv_out1, ocv_out2, ocv_essentialmat, ocv_inliers);

    //SVD essentialmat
    std::vector<cv::Mat> ocv_Rs, ocv_ts;
    getRt(ocv_essentialmat, ocv_Rs, ocv_ts);
    
    return 0;
}
