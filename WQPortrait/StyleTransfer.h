#pragma once
#include "commerHeader.h"
#include "PerlinNoise.h"

class StyleTransfer
{
public:
	StyleTransfer();
	~StyleTransfer();
	cv::Mat nearColorEdgeDetector(cv::Mat src);
	cv::Mat largeGradient(cv::Mat src);
	cv::Mat edgeTypeDetect(cv::Mat src, cv::Mat edge, cv::Mat gradients, cv::Mat mask);
	cv::Mat wetPre(cv::Mat input, cv::Mat edgeType, cv::Mat gradients, cv::Mat randomNoiseImage);
	cv::Mat wetFilter(cv::Mat wetpre, cv::Mat gradients);
	cv::Mat edgeDarken(cv::Mat distortImg, cv::Mat wetfilterImg, cv::Mat mask, double level, int flag);
	cv::Mat pigmentVariation(cv::Mat input, cv::Mat randomNoiseImg, cv::Mat perlinNoiseImage, cv::Mat paper, cv::Mat mask, double level, int flag);
	cv::Mat boundaryDistort(cv::Mat input, cv::Mat mask, cv::Mat perlinNoiseImage, int is_white, int flag);//input:CV_32FC3,output:CV_32FC3
};

