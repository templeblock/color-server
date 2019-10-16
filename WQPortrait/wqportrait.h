#ifndef WQPORTRAIT_H
#define WQPORTRAIT_H
#pragma once
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QDialog>  
#include <qlabel.h>
#include <qfiledialog.h>
#include <qscrollarea.h>
#include <Windows.h>
#include <iostream>
#include <fstream>
#include "ui_wqportrait.h"
#include "StyleTransfer.h"
#include "Tool.h"


class WQPortrait : public QMainWindow
{
	Q_OBJECT

public:
	WQPortrait(QWidget *parent = 0);
	~WQPortrait();
	float getDistance(cv::Vec3f a, cv::Vec3f b);
	void saliency(cv::Mat face);
	void forward_backward_watercolor(std::vector<cv::Mat> frames, std::vector<int> keyframe_index, std::vector<cv::Mat>& watercolor_frames);
	void dry_brush();
	void adjustColor(cv::Mat frame, cv::Mat& output);

private:
	Ui::WQPortraitClass ui;
	QLabel* label_result;
	QLabel* label_src;
	QImage* img;
	std::string file_name, flow_path;
	cv::Mat randomNoiseImage, perlinNoiseImage, paper;
	cv::Mat src, src_color;
	cv::Mat first_frame;
	cv::Mat face_mask, abstMat, abstMat_4f, wetfilter, darken, boundDistort, pigment, error_mask,pigment_dry;
	int flag;
	cv::VideoCapture cap;
	bool is_video;

private slots:
	void open();
	void abstraction_level(int level);
	void wet_level(int level);
	void darken_level(double level);
	void test();
	void is_white(int index);
	void show_result(int index);
	void pigment_level(double level);
	void is_dry(int index);
	void video_process();
	void image_process();
};

#endif // WQPORTRAIT_H
