#include "stdio.h"
#include "time.h"

#include <iostream>
#include <fstream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"

#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "PnPSolver.hpp"
#include "PCLView.hpp"
#include "RotationTools.hpp"
#include "Triangulation.hpp"
#include "5point.h"

using namespace std;
using namespace cv;

class FeatureFrame
{
    public:
        cv::Mat frame;
        cv::Mat frame_gray;
        cv::Mat descriptor; 
        std::vector<cv::KeyPoint> keypoints;
        FeatureFrame(){};
        FeatureFrame(cv::Mat &f, ORBextractor &extractor){
            f.copyTo(frame);
            cv::cvtColor(f, frame_gray, CV_BGR2GRAY);
            extractor(frame_gray, cv::Mat(), keypoints, descriptor);
        };
        ~FeatureFrame(){};
};

class ImageCapture
{
    public:

        ifstream fs;
        string imagename;
        string dirname;
        
        ImageCapture(){
            dirname = "";
        };
        
        ImageCapture(string listname){
            open(listname);
        };

        ImageCapture(string listname, string dirname){
            open(listname, dirname);
        };

        void open(string listname){
            dirname = "";
            fs.open(listname);
        };

        void open(string listname, string dirname){
            dirname = dirname;
            fs.open(listname);
        };

        bool isOpened(){
          return true; 
        };

        bool read(cv::Mat &frame){
            getline(fs, imagename);
            frame = imread(dirname+imagename);
            return !frame.empty();
        };
};

class HybridCapture
{
    public:

        ImageCapture imageCapture;
        cv::VideoCapture videoCapture;
        int capture_type;

        void open(string name, int type){
           capture_type = type; 
           if(capture_type==0){
                videoCapture.open(name);
           }else{
               imageCapture.open(name);
           }
        };

        void open(string listname, string dirname){
            capture_type = 1;
           imageCapture.open(listname, dirname); 
        };

        bool isOpened(){
            if(capture_type==0){
                return videoCapture.isOpened();
            }else{
                return imageCapture.isOpened();
            }
        };

        bool read(cv::Mat &frame){
            if(capture_type==0){
                return videoCapture.read(frame);
            }else{
                return imageCapture.read(frame);
            }
        };
};

class MatchRelation
{

    public:
        FeatureFrame *ff1, *ff2;
        std::vector<int> RIndex;
        std::vector<int> LIndex;
        cv::Mat inliers;
        cv::Mat R, t;

};

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
		std::vector<cv::Point2f>& srcKeyPoints, std::vector<cv::Point2f>& desKeyPoints,
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
		cv::Point2f* kpt1 = &srcKeyPoints[t];
		cv::Point2f* kpt2 = &desKeyPoints[t];
		cv::Point pt1(kpt1->x, kpt1->y);
		cv::Point pt2(kpt2->x + src.cols, kpt2->y);
		cv::Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawLink(resultMat, pt1, pt2, color);
        num_match++;
	}
    printf("num match: %d\n", num_match);

	cv::namedWindow(win_name);
	cv::imshow(win_name ,resultMat);

	//while(cv::waitKey()<0){}

}

void getEssentialMat(cv::Mat &cameraMatrix,std::vector<cv::KeyPoint*> &out1, std::vector<cv::KeyPoint*> &out2, cv::Mat &result, cv::Mat &inliers)
{
    //calculate fundamentalmat and essentialmat
    std::vector<cv::Point2f> ori_pt;
    std::vector<cv::Point2f> des_pt;
    cv::Mat fundamentalmat;
    
    
    for(int i = 0, _end = out1.size(); i<_end; i++){
        ori_pt.push_back(out1[i]->pt);
        des_pt.push_back(out2[i]->pt);
    }
    //PnPSolver::FindFundamentalMatRansac(ori_pt, des_pt, fundamentalmat, inliers);
    //cout << "cv fundamentalMat" << endl << fundamentalmat << endl; 
    //
    std::vector<cv::Mat> fundamentalMats, projectMats, inlierMats;
    std::vector<int> inlierNums;
    Solve5PointEssential(ori_pt, des_pt, fundamentalMats, projectMats, inlierNums, inlierMats);
    int maxInlier = -1;
    int best_idx = -1;
    for(int i = 0, _end = fundamentalMats.size(); i< _end;i++){
        if(inlierNums[i] > maxInlier){
            maxInlier = inlierNums[i];
            best_idx = i;
        }
    }
    fundamentalmat = fundamentalMats[best_idx];
    inliers = inlierMats[best_idx];
    cout << "5point fundamentalmat" << endl << fundamentalmat << endl;

    PnPSolver::FindEssentialMat(fundamentalmat, cameraMatrix, result);
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
    cv::SVD::compute(essentialmat, w, u, vt, SVD::FULL_UV);

    Rs.push_back(u * WMat * vt);
    Rs.push_back(u * WMat.t() * vt);

    ts.push_back(u.col(2));
    ts.push_back(-u.col(2));

    //std::cout << "R1: " << std::endl << Rs[0] << std::endl;
    //std::cout << "R2: " << std::endl << Rs[1] << std::endl;
    //std::cout << "t1: " << std::endl << ts[0] << std::endl;
    //std::cout << "t1: " << std::endl << ts[1] << std::endl;
    
}

void getPointsFromHomogeneous(cv::Mat &h, std::vector<cv::Point3f> &pts){

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


void triangulate(
        cv::Mat &cameraMatrix,
        std::vector<cv::Point2f> &f1_pt, 
        std::vector<cv::Point2f> &f2_pt, 
        cv::Mat R, 
        cv::Mat t, 
        std::vector<cv::Point3f> &result){
    
    cv::Mat p4s;

    double projArray1[12] = 
    {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    }; 
    cv::Mat projMat1(3, 4, CV_64FC1, projArray1);
    projMat1 = cameraMatrix * projMat1;

    cv::Mat projMat2(3,4,CV_64FC1); 

    R.copyTo(projMat2(cv::Rect(0,0,3,3)));
    t.copyTo(projMat2(cv::Rect(3,0,1,3)));

    cv::triangulatePoints(projMat1, cameraMatrix*projMat2, f1_pt, f2_pt, p4s);

    cout << "[DEBUG]" << getImageType(p4s) << endl;
    cout << "[DEBUG]" << endl << p4s << endl;

    getPointsFromHomogeneous(p4s, result);

}

MatchRelation computePose(cv::Mat &cameraMatrix, FeatureFrame &f1, FeatureFrame &f2, cv::Mat &R, cv::Mat &t)
{
    double fx = cameraMatrix.at<double>(0, 0),
           fy = cameraMatrix.at<double>(1, 1),
           cx = cameraMatrix.at<double>(0, 2),
           cy = cameraMatrix.at<double>(1, 2);

    MatchRelation relation;
    relation.ff1 = &f1;
    relation.ff2 = &f2;

    //orb feature match
    ORBmatcher orb_matcher(0.9, true);
    orb_matcher.SearchByKeypointsInGrid(
            f1.frame.cols, 
            f1.frame.rows, 
            f1.keypoints, f1.descriptor, f2.keypoints, f2.descriptor, 
            relation.LIndex, relation.RIndex);

    //get essentialmat
    cv::Mat essentialmat;
    cv::Mat inliers;
    //get matched points
    std::vector<cv::Point2f> f1_pt, f2_pt;
    for(int i= 0 ,_end = relation.LIndex.size(); i < _end; i++){
            f1_pt.push_back(f1.keypoints[relation.LIndex[i]].pt);
            f2_pt.push_back(f2.keypoints[relation.RIndex[i]].pt);
    }
    //getEssentialMat(cameraMatrix, out1, out2, essentialmat,inliers);
    essentialmat = cv::findEssentialMat(f1_pt, f2_pt, fx/2 + fy/2, cv::Point2d(cx, cy), RANSAC, 0.999, 1.0, inliers);

    inliers.copyTo(relation.inliers);

    //draw match to debug
    drawMatchResult("match", f1.frame, f2.frame, f1_pt, f2_pt, inliers);
    //cv::waitKey(-1);
    
    //recover pose
    cv::recoverPose(essentialmat, f1_pt, f2_pt, R, t, fx/2 + fy/2, cv::Point2d(cx, cy), inliers);
    R.copyTo(relation.R);
    t.copyTo(relation.t);

    return relation;

    //SVD the essentialmat
    //std::vector<cv::Mat> Rs, ts;
    //getRt(essentialmat, Rs, ts); 

    //get matched inliers points
    //std::vector<cv::Point2f> f1_pt, f2_pt;
    //for(int i= 0 ,_end = inliers.rows; i < _end; i++){
    //    if(inliers.data[i]){
    //        f1_pt.push_back(out1[i]->pt);
    //        f2_pt.push_back(out2[i]->pt);
    //    }
    //}

    ////triangulate
    //std::vector<cv::Point3f> pts_3d;
    //int R_idx = -1 , t_idx = -1;
    //triangulate(cameraMatrix, f1_pt, f2_pt, Rs, ts, pts_3d, R_idx, t_idx);

    //if(R_idx>=0&&t_idx>=0){
    //    R = Rs[R_idx];
    //    t = ts[t_idx];
    //    return true;
    //}else{
    //    return false;
    //}
    
}

double ComputeRelativeScale(cv::Mat &cameraMatrix, 
        MatchRelation &relation1, 
        MatchRelation &relation2){
    std::vector<cv::Point2f> l_pt, m_pt, r_pt;
    std::vector<cv::Point3f> l_pt3d, r_pt3d;

    int num_pts = 0;

    for(int i = 0, _end = relation1.LIndex.size(); i < _end && num_pts < 8; i++){
        if(relation1.inliers.data[i]){
            int r_base = relation1.RIndex[i];
            for(int j = 0, _endj = relation2.LIndex.size(); j < _endj; j++){
                if(relation2.LIndex[j] == r_base && relation2.inliers.data[j]){
                    l_pt.push_back(relation1.ff1->keypoints[relation1.LIndex[i]].pt);
                    m_pt.push_back(relation2.ff1->keypoints[relation2.LIndex[j]].pt);
                    r_pt.push_back(relation2.ff2->keypoints[relation2.RIndex[j]].pt);
                    num_pts++;
                    i+=10;
                    break;
                }
            }
        }
    }

    cout << "[DEBUG] " << relation1.ff2 << "  " << relation2.ff1 << endl;  

    drawMatchResult("l and m", relation1.ff1->frame, relation1.ff2->frame, l_pt, m_pt);
    drawMatchResult("m and r", relation2.ff1->frame, relation2.ff2->frame, m_pt, r_pt);

    if(num_pts < 4){
        cout << "warning: not enough points to compute relative scale" << endl;
        return 1;
    }

    triangulate(cameraMatrix, l_pt, m_pt, relation1.R, relation1.t, l_pt3d);
    triangulate(cameraMatrix, m_pt, r_pt, relation2.R, relation2.t, r_pt3d);

    cout << "[DEBUG] " << l_pt3d.size()  << "  "<< r_pt3d.size() << endl;
    cout << "[DEBUG] " << endl << l_pt3d[0] << l_pt3d[1] << l_pt3d[2] << l_pt3d[3] << endl 
        << r_pt3d[0] << r_pt3d[1] << r_pt3d[2] << r_pt3d[3] << endl;

    double scale = 0;
    int count = 0;

    for(int i = 0, _endi = l_pt3d.size(); i < _endi; i++){
        for(int j = i+1; j < _endi; j++){
            double r_dx = r_pt3d[i].x - r_pt3d[j].x, 
                   r_dy = r_pt3d[i].y - r_pt3d[j].y, 
                   r_dz = r_pt3d[i].z - r_pt3d[j].z;
            double l_dx = l_pt3d[i].x - l_pt3d[j].x, 
                   l_dy = l_pt3d[i].y - l_pt3d[j].y, 
                   l_dz = l_pt3d[i].z - l_pt3d[j].z;
            scale = scale + ((sqrt(r_dx*r_dx+r_dy*r_dy+r_dz*r_dz))/(sqrt(l_dx*l_dx+l_dy*l_dy+l_dz*l_dz)));
            count++;
        }
    }

    cout << "[DEBUG] scale sum:" << scale << " count " << count << endl;

    return scale / count;

}

int main(int argc, char** argv )
{
    if(argc < 5){
        printf("need more argument\n");
        return 0;
    }

    string camerafileName, dirname = "", filename;
    int capture_type = -1;

    for(int i = 1; i< argc; i+=2){
        if(strcmp("-c", argv[i])==0){
            camerafileName = std::string(argv[i+1]); 
        }else
        if(strcmp("-d", argv[i])==0){
            dirname = std::string(argv[i+1]); 
        }else
        if(strcmp("-f", argv[i])==0){
            filename = std::string(argv[i+1]);
            capture_type = 1; 
        }else 
        if(strcmp("-v", argv[i])==0){
            filename = std::string(argv[i+1]);
            capture_type = 0; 
        }
    }

    cv::namedWindow("video");
    
    double fx, fy, cx, cy;

    std::ifstream fs(camerafileName);
    std::string line;
    std::getline(fs, line);
    std::stringstream ss(line);
    ss >> fx >> fy >> cx >> cy ;
    cout << fx << " " << fy << "  " << cx << "  " << cy << endl;
    
    //samsung camera
    //double fx= 528.253;
    //double fy= 531.901;
    //double cx= 323.617;
    //double cy= 235.532;

    //logitech camera
    //double fx= 616.662852;
    //double fy= 618.719985;
    //double cx= 323.667190;
    //double cy= 236.399432;

    //freiburg 2
    //double fx= 520.9;
    //double fy= 521.0;
    //double cx= 325.1;
    //double cy= 249.7;
    double cameraArray[9] = 
    {
        fx, 0 , cx,
        0,  fy, cy,
        0,  0,  1
    };
    cv::Mat cameraMatrix(3, 3, CV_64FC1, cameraArray);

    //map to show
    cv::Mat map1(1000, 1000, CV_8UC3, cv::Scalar::all(255)),
        map2(1000, 1000, CV_8UC3, cv::Scalar::all(255)),
        map3(1000, 1000, CV_8UC3, cv::Scalar::all(255));

    double x , y , z;
    x = y = z = 500;
    
    //orb feature extractor
    ORBextractor orb_extractor(1000, 1.2, 8, 20, 7);

    //capture video or image sequence
    //cv::VideoCapture capture;
    //ImageCapture capture;
    HybridCapture capture;
    if(dirname.compare("")!=0){
        capture.open(filename, dirname);
    }else{
        capture.open(filename, capture_type);
    }
    if(!capture.isOpened())
    {
        printf("video open failed! %s\n", argv[1]);
        return 0;
    }
    cv::Mat videoFrame;
    FeatureFrame FrameQ[3];
    MatchRelation RelationQ[2];

    int l_f_idx, m_f_idx, r_f_idx;
    int l_r_idx, r_r_idx;

    capture.read(videoFrame);
    cv::imshow("video", videoFrame);
    cv::waitKey(-1);
    FrameQ[0]= FeatureFrame(videoFrame, orb_extractor);

    long index = 0;
    cv::Mat R, t, g_t(3, 1, CV_64FC1, cv::Scalar::all(0));
    double r_scale = 1;
    while(capture.read(videoFrame)){
        cout << endl;
        index++;
        
        l_f_idx = (index-2) % 3;
        m_f_idx = (index-1) % 3;
        r_f_idx = index % 3;

        l_r_idx = (index-1) % 2;
        r_r_idx = index % 2;

        cout << "l_f_idx:" << l_f_idx << "m_f_idx:" << m_f_idx << "r_f_idx:" << r_f_idx << endl;
        cout << "l_r_idx:" << l_r_idx << "r_r_idx:" << r_r_idx << endl;
        FrameQ[r_f_idx] = FeatureFrame(videoFrame, orb_extractor);
        RelationQ[r_r_idx] = computePose(cameraMatrix,FrameQ[m_f_idx], FrameQ[r_f_idx], R, t);
        //compute relative scale
        if(index > 1){
            r_scale = ComputeRelativeScale(cameraMatrix, RelationQ[l_r_idx], RelationQ[r_r_idx]); 
            cout << "relative scale:" << r_scale << endl;
        }

        cout << "translate: x=" << t.at<double>(0) << " y=" << t.at<double>(1) << " z= " << t.at<double>(2) << endl;
        g_t = R*g_t + r_scale * t;
        cout << "global translate: x=" << g_t.at<double>(0) << " y=" << g_t.at<double>(1) << " z= " << g_t.at<double>(2) << endl;
        double x_ = 500 + g_t.at<double>(0)*1;
        double y_ = 500 + g_t.at<double>(1)*1;
        double z_ = 500 + g_t.at<double>(2)*1;
        cv::line(map1, Point2f(x, y), Point2f(x_, y_), cv::Scalar(255, 0, 0) );
        cv::line(map2, Point2f(y, z), Point2f(y_, z_), cv::Scalar(0, 255, 0) );
        cv::line(map3, Point2f(z, x), Point2f(z_, x_), cv::Scalar(0, 0, 255) );
        x = x_; y = y_; z = z_;
        cv::namedWindow("map1");
        cv::namedWindow("map2");
        cv::namedWindow("map3");
        cv::imshow("map1", map1);
        cv::imshow("map2", map2);
        cv::imshow("map3", map3);
        //cv::waitKey(1);

        cv::waitKey( -1);

    }

    cv::waitKey(-1);

    ////orb feature match
    //std::vector<cv::KeyPoint*> out1, out2;
    //ORBmatcher orb_matcher(0.9, true);
    //time = clock();
    //orb_matcher.SearchByKeypointsInGrid(
    //        frame1.cols, 
    //        frame1.rows, 
    //        kp1, dc1, kp2, dc2, 
    //        out1, out2);
    //printf("time match: %f\n", (clock()-time)/(double)CLOCKS_PER_SEC);

    ////draw orb result
    ////drawMatchResult("orb result", frame1, frame2, out1, out2);

    ////get essentialmat
    //cv::Mat essentialmat;
    //cv::Mat inliers;
    //getEssentialMat(frame1, frame2, out1, out2, essentialmat,inliers);

    ////SVD the essentialmat
    //std::vector<cv::Mat> Rs, ts;
    //getRt(essentialmat, Rs, ts); 

    //cout << "Rs0" << endl << rotationMatrixToEulerAngles(Rs[0]) << endl;
    //cout << "Rs1" << endl << rotationMatrixToEulerAngles(Rs[1]) << endl;

    ////get matched inliers points
    //std::vector<cv::Point2f> f1_pt, f2_pt;
    //for(int i= 0 ,_end = inliers.rows; i < _end; i++){
    //    if(inliers.data[i]){
    //        f1_pt.push_back(out1[i]->pt);
    //        f2_pt.push_back(out2[i]->pt);
    //    }
    //}
    //cout << "matched inliers points: " << f1_pt.size() << endl;

    ////triangulate
    //std::vector<cv::Point3f> pts_3d;
    //triangulate(f1_pt, f2_pt, Rs, ts, pts_3d);
    //
    //PCLView view;
    //view.showRGBPoints(frame1, f1_pt, pts_3d);

    ////////////////////////////////////////////////////////////
    //opencv feature extract
    //cv::Ptr<cv::FeatureDetector> _detector;
    //cv::Ptr<cv::DescriptorExtractor> _descriptor;
    //cv::initModule_nonfree();
    //_detector = cv::FeatureDetector::create("GridSIFT");
    //_descriptor = cv::DescriptorExtractor::create("SIFT");

    //std::vector<cv::KeyPoint> ocv_kp1, ocv_kp2;
    //cv::Mat desp1, desp2;

    //time = clock();
    //_detector->detect(frame1, ocv_kp1);
    //_detector->detect(frame2, ocv_kp2);
    //_descriptor->compute(frame1, ocv_kp1, desp1);
    //_descriptor->compute(frame2, ocv_kp2, desp2);
    //printf("time extract feature opencv: %f\n", (clock()-time)/(double)CLOCKS_PER_SEC);

    ////opencv feature match
    //std::vector<cv::DMatch> matches;
    //cv::FlannBasedMatcher matcher;
    //time = clock();
    //matcher.match(desp1, desp2, matches);

    ////filter good matched
    //std::vector< cv::DMatch > goodMatches;
    //double minDis = 9999, maxDis = -1;
    //for ( size_t i=0; i<matches.size(); i++ )
    //{
    //    if ( matches[i].distance < minDis )
    //        minDis = matches[i].distance;
    //    if(matches[i].distance > maxDis)
    //        maxDis = matches[i].distance;
    //}

    //for ( size_t i=0; i<matches.size(); i++ )
    //{
    //    if (matches[i].distance < 0.25*(maxDis - minDis) + minDis)
    //        goodMatches.push_back( matches[i] );
    //} 
    //printf("time match feature opencv: %f\n", (clock()-time)/(double)CLOCKS_PER_SEC);
    //
    ////draw opencv result
    //cv::Mat imageMatches;
    //cv::drawMatches(frame1, ocv_kp1, frame2, ocv_kp2, goodMatches, imageMatches);
    //cv::namedWindow("opencv result");
    //cv::imshow("opencv result", imageMatches);
    //while(cv::waitKey()<0){}

    ////get essentialmat
    //std::vector<cv::KeyPoint*> ocv_out1;
    //std::vector<cv::KeyPoint*> ocv_out2;

    //for(int i = 0, _end = goodMatches.size(); i<_end; i++ ){
    //    ocv_out1.push_back(&ocv_kp1[goodMatches[i].queryIdx]);
    //    ocv_out2.push_back(&ocv_kp2[goodMatches[i].trainIdx]);
    //}

    //cv::Mat ocv_essentialmat;
    //cv::Mat ocv_inliers;
    //getEssentialMat(frame1, frame2, ocv_out1, ocv_out2, ocv_essentialmat, ocv_inliers);

    ////SVD essentialmat
    //std::vector<cv::Mat> ocv_Rs, ocv_ts;
    //getRt(ocv_essentialmat, ocv_Rs, ocv_ts);

    //cout << "ocv_Rs0" << endl << rotationMatrixToEulerAngles(ocv_Rs[0]) << endl;
    //cout << "ocv_Rs1" << endl << rotationMatrixToEulerAngles(ocv_Rs[1]) << endl;

    ////get matched inliers points
    //std::vector<cv::Point2f> ocv_f1_pt, ocv_f2_pt;
    //for(int i= 0 ,_end = ocv_inliers.rows; i < _end; i++){
    //    if(ocv_inliers.data[i]){
    //        ocv_f1_pt.push_back(ocv_out1[i]->pt);
    //        ocv_f2_pt.push_back(ocv_out2[i]->pt);
    //    }
    //}
    //
    ////triangulate
    //std::vector<cv::Point3f> ocv_pts_3d;
    //triangulate(ocv_f1_pt, ocv_f2_pt, ocv_Rs, ocv_ts, ocv_pts_3d);
    //
    //view.showRGBPoints(frame1, ocv_f1_pt, ocv_pts_3d);
    
    return 0;
}
