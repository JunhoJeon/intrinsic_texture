#include <ANN/ANN.h>
//#ifdef _CHAR16T
//#define CHAR16_T
//#endif
#include "mex.h"
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
void LLE(cv::Mat1f &X, cv::Mat1f &W, cv::Mat1i &neighbors, int N, int D, int K);
void LLE(cv::Mat1f &X, cv::Mat1f &W, cv::Mat1i &neighbors, int N, int D, int K, int wSize, int width, int height);
void LLE(cv::Mat1f &X, cv::Mat1f &vMap, cv::Mat1f &W, cv::Mat1i &neighbors, int N, int D, int K, int wSize, int width, int height, float varThres);
void LLE3(cv::Mat1f &X, cv::Mat1f &W, cv::Mat1i &neighbors, int N, int D, int K);