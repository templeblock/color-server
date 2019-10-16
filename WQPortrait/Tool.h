#pragma once
#include "commerHeader.h"
#include <qimage.h>
#include <qdebug.h>
//using namespace cv;

class Tool
{
public:
	Tool();
	~Tool();
	cv::Mat GaussianBlur(cv::Mat f, float sigma);
	cv::Mat QImage2cvMat(QImage image);
	QImage cvMat2QImage(const cv::Mat& mat);

private:
	cv::Mat convolveVertical1D(cv::Mat f, float* kernel, int k_len);
	cv::Mat convolveHorizontal1D(cv::Mat f, float* kernel, int k_len);
	
};

