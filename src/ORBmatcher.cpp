/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<stdint-gcc.h>

using namespace std;

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

int ORBmatcher::SearchByKeypointsInGrid(
        int width, int height,
        std::vector<cv::KeyPoint> &keypoints1, 
        cv::Mat &descriptors1, 
        std::vector<cv::KeyPoint> &keypoints2, 
        cv::Mat &descriptors2,
        std::vector<int> &out1, 
        std::vector<int> &out2)
{
    //build grid
    int g_s = width/16.0;
    int g_n_w = width/g_s + 1;
    int g_n_h = height/g_s + 1;
    vector< vector<int> > grids1(g_n_w*g_n_h);
    vector< vector<int> > grids2(g_n_w*g_n_h);
    int np1 = descriptors1.rows, np2 = descriptors2.rows;
    for(int i = 0; i < np1; i++){
        cv::KeyPoint *kp = &keypoints1[i];
        grids1[kp->pt.y/g_s * g_n_w + kp->pt.x/g_s].push_back(i);
    }
    for(int i = 0; i < np2; i++){
        cv::KeyPoint *kp = &keypoints2[i];
        grids2[kp->pt.y/g_s * g_n_w + kp->pt.x/g_s].push_back(i);
    }

    //match in grid
    int num_match = 0;
    for(size_t I = 0, _end = grids1.size(); I < _end; I++)
    {
        int g_x = (int)I % g_n_w;
        int g_y = (int)I / g_n_w; 

        for(vector<int>::iterator iter = grids1[I].begin(), iter_end = grids1[I].end(); iter != iter_end; iter++){
            int kp1_idx = *iter; 
            int min_dis = 9999, sec_min_dis = 9999;
            int index = 0, sec_index = 0;

            for(int g_i = g_x-2; g_i <= g_x+2; g_i++){
                for(int g_j = g_y-2; g_j <= g_y+2; g_j++){
                    int g_I = g_j * g_n_w + g_i;
                    if(g_I < 0 || g_I >= (int)_end)
                        continue;
                    for(vector<int>::iterator _iter = grids2[g_I].begin(), _iter_end = grids2[g_I].end(); _iter != _iter_end; _iter++){
                        int kp2_idx = *_iter;
                        if( abs(keypoints2[kp2_idx].angle-keypoints1[kp1_idx].angle) > 100 )
                            continue;
                        int distance = ORBmatcher::DescriptorDistance(
                                descriptors1.row(kp1_idx),
                                descriptors2.row(kp2_idx));
                        if(distance < min_dis){
                            sec_min_dis = min_dis;
                            sec_index = index;
                            min_dis = distance;
                            index = kp2_idx;
                        }else if(distance < sec_min_dis){
                            sec_min_dis = distance;
                            sec_index = kp2_idx;
                        }
                    } 
                }
            }

            if(min_dis<=ORBmatcher::TH_LOW && min_dis <= mfNNratio * sec_min_dis){
                out1.push_back(kp1_idx);
                out2.push_back(index);
                num_match++;
            }
        }
    }
    return num_match;

}

int ORBmatcher::SearchByKeypoints(
        std::vector<cv::KeyPoint> &keypoints1, 
        cv::Mat &descriptors1, 
        std::vector<cv::KeyPoint> &keypoints2, 
        cv::Mat &descriptors2,
        std::vector<cv::KeyPoint*> &out1, 
        std::vector<cv::KeyPoint*> &out2)
{
    int np1 = descriptors1.rows;
    int np2 = descriptors2.rows;

    int dis[1000][1000];

    float distance = 0;
    int index = 0;

    for(int i = 0; i < np1; i++){
        for(int j = 0; j < np2; j++){
            dis[i][j] = ORBmatcher::DescriptorDistance(descriptors1.row(i), descriptors2.row(j));
        }
    }

    vector<int> map1;
    for(int i = 0; i < np1; i++){
        float min_dis = 9999;
        for(int j = 0; j < np2; j++){
            distance = dis[i][j];
            if(distance<min_dis){
                min_dis = distance;
                index = j;
            } 
        }
        if(min_dis <= ORBmatcher::TH_LOW)
            map1.push_back(index);
        else
            map1.push_back(-1);
    }

    vector<int> map2;
    for(int i = 0; i < np2; i++){
        float min_dis = 9999;
        for(int j = 0; j < np1; j++){
            distance = dis[j][i];
            if(distance<min_dis){
                min_dis = distance;
                index = j;
            } 
        }
        if(min_dis <= ORBmatcher::TH_LOW)
            map2.push_back(index);
        else
            map2.push_back(-1);
    }

    int num_match = 0;
    if(np1 < np2){
        for(int i = 0; i<np1; i++ ){
            if(map1[i] && map2[map1[i]]==i){
                out1.push_back(&keypoints1[i]);
                out2.push_back(&keypoints2[map1[i]]);
                num_match++;
            }
        }
    }else{
        for(int i = 0; i<np2; i++ ){
            if( map2[i] && map1[map2[i]]==i){
                out2.push_back(&keypoints2[i]);
                out1.push_back(&keypoints1[map2[i]]);
                num_match++;
            }
        }
    }
    return num_match;
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
