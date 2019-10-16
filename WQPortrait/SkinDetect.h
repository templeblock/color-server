#pragma once
#include "commerHeader.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>

using namespace dlib;
using namespace std;

class SkinDetect
{
public:
	cv::Mat img;
	SkinDetect(cv::Mat src);
	~SkinDetect();
	int rows;
	int cols;
	bool is_resize;

	bool face_detect(cv::Mat &dst);

	
};

