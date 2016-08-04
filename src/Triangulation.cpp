#include "Triangulation.hpp"

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

