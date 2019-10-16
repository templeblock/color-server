#pragma once
#include <stdlib.h>
#include <Math.h>
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

class FaceDeformation
{
public:
	FaceDeformation();
	~FaceDeformation();
	void cheek_deformation(cv::Mat src, dlib::full_object_detection shape, int level);
	void eyes_deformation(cv::Mat &src, dlib::full_object_detection shape, int level);
	void nose_deformation(cv::Mat src, dlib::full_object_detection shape, int level);
	void mouth_deformation(cv::Mat src, dlib::full_object_detection shape, int level);
	void deformation(cv::Mat &src);
};

