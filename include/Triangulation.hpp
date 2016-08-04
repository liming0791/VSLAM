#include "stdio.h"
#include "time.h"

#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//Triagulate points
void TriangulatePoints(const vector<Point2f>& pt_set1,
                       const vector<Point2f>& pt_set2,
                       const Matx33f& K,
                       const Matx34f& P,
                       const Matx34f& P1,
                       vector<Point3f>& pointcloud);
