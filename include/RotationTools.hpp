#include "stdio.h"
#include "time.h"

#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f rotationMatrixToEulerAngles(Mat &R);
