#include "wqportrait.h"

extern "C" void wet_main_cuda(cv::Mat abst, cv::Mat& dst, cv::Mat perlin_mat, int level, cv::Mat mask, int flag);
//extern "C" void abstract_cuda(cv::Mat src, cv::Mat& dst, int radius, int nBins, cv::Mat mask);

WQPortrait::WQPortrait(QWidget *parent)
	: QMainWindow(parent)
{
	flag = 0;
	is_video = false;
	randomNoiseImage = cv::imread("data/randomnoise.png");
	perlinNoiseImage = cv::imread("data/perlinnoise.png");
	paper = cv::imread("data/paper.jpg");
	//is_portrait = false;
	label_result = new QLabel(this);
	label_src = new QLabel(this);
	img = new QImage;
	ui.setupUi(this);

}

WQPortrait::~WQPortrait()
{
	if (label_result != nullptr)
	{
		delete label_result;
		label_result = nullptr;
	}
	if (label_src != nullptr)
	{
		delete label_src;
		label_src = nullptr;
	}
	if (img != nullptr)
	{
		delete img;
		img = nullptr;
	}
		
}

/***********************************************************************
*	打开菜单响应函数
*   可以是视频或图像，视频会显示第一帧
***********************************************************************/
void WQPortrait::open()
{
	QString filename = QFileDialog::getOpenFileName(this,
		tr("Open"),
		"",
		tr("Image Video(*.png *.bmp *.jpg *.mp4)"));
	file_name = filename.toStdString();
	QPixmap mypixmap;
	
	if (filename.isEmpty())
	{
		QMessageBox::information(this, tr("Empty File"), tr("Empty File!"));
		return;
	}
	else{
		//判断是图像还是视频
		int p = file_name.find(".");
		std::string suffix = file_name.substr(p);//获取后缀
		is_video = false;
		if (suffix == ".jpg" || suffix == ".png" || suffix == ".bmp" || suffix == ".JPG" || suffix == ".PNG" || suffix == ".BMP")
		{
			if (!(img->load(filename)))
			{
				QMessageBox::information(this, tr("Open Failed"), tr("Open Failed!"));
				delete img;
				img = nullptr;
				return;
			}
			mypixmap = QPixmap::fromImage(*img);
			
		}
		else{
			cap.open(filename.toStdString());
			if (!cap.isOpened())
			{
				QMessageBox::critical(0, "critical message", "can't open video!", QMessageBox::Ok | QMessageBox::Escape, 0);
				return;
			}
			cap.read(first_frame);

			cv::Mat first_frame_show;
			cvtColor(first_frame, first_frame_show, CV_BGR2RGB);
			const uchar *pSrc = (const uchar*)first_frame_show.data;

			QImage qimage(pSrc, first_frame_show.cols, first_frame_show.rows, first_frame_show.step, QImage::Format_RGB888);
			mypixmap = QPixmap::fromImage(qimage);

			
		}
		//显示原图像
		
		label_src->setPixmap(mypixmap);
		label_src->resize(label_src->pixmap()->size());
		label_src->setAlignment(Qt::AlignCenter);

		ui.scrollArea_2->setWidget(label_src);
		ui.scrollArea_2->setAlignment(Qt::AlignCenter);
		ui.scrollArea_2->setWidgetResizable(true);  // 自动调整大小
		ui.scrollArea_2->show();

		//mypixmap2 = mypixmap;
		label_result->setPixmap(mypixmap);
		label_result->resize(label_result->pixmap()->size());
		label_result->setAlignment(Qt::AlignCenter);

		ui.scrollArea->setWidget(label_result);
		ui.scrollArea->setAlignment(Qt::AlignCenter);
		ui.scrollArea->setWidgetResizable(true);  // 自动调整大小
		ui.scrollArea->show();

		if (!(suffix == ".jpg" || suffix == ".png" || suffix == ".bmp" || suffix == ".JPG" || suffix == ".PNG" || suffix == ".BMP"))
		{
			first_frame.copyTo(src);
			saliency(src);
			adjustColor(src, src_color);
			abstraction_level(ui.spinBox_abst->value());
		}
	}//file is not empty
}


/***********************************************************************
*	显著性物体检测，直接调用，速度比较快
*   参考论文：Minimum Barrier Salient Object Detection at 80 FPS
***********************************************************************/
void WQPortrait::saliency(cv::Mat face)
{
	imwrite("s/a.jpg", face);
	std::string file_path = "MBS.exe ./s/ ./image/ 1";
	int res = WinExec(file_path.data(), SW_HIDE);
	Sleep(200);//睡眠是因为exe运行慢，有时读取的还是原来的文件

	cv::Mat saliency = cv::imread("image/a_MB+.png", cv::IMREAD_GRAYSCALE);
	
	saliency.copyTo(face_mask);
}


float WQPortrait::getDistance(cv::Vec3f a, cv::Vec3f b)
{
	return sqrt(pow(a[0] - b[0], 2.0) + pow(a[1] - b[1], 2.0) + pow(a[2] - b[2], 2.0));
}


/***********************************************************************
*	抽象简化参数调节响应函数
***********************************************************************/
void WQPortrait::abstraction_level(int level)
{
	if (src.empty())
	{
		QMessageBox::critical(0, "critical message", "Please open an image!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	
	//fast global image smoothing
	cv::Mat abstMat_3f_fine, abstMat_3f_coarse;
	cv::Vec3f tempa;
	double labmda = level*100;
	double color_sigma = level/2;
	cv::ximgproc::fastGlobalSmootherFilter(src_color, src_color, abstMat_3f_fine, labmda, color_sigma, 0.3, 2);
	cv::ximgproc::fastGlobalSmootherFilter(src_color, src_color, abstMat_3f_coarse, (level +8) * 100, level+4, 0.3, 2);//4,2
	
	abstMat_3f_fine.convertTo(abstMat_3f_fine, CV_32FC3, 1 / 255.0);
	abstMat_3f_coarse.convertTo(abstMat_3f_coarse, CV_32FC3, 1 / 255.0);
	
	abstMat.create(src_color.rows, src_color.cols, CV_32FC3);
	for (int i = 0; i < abstMat_3f_coarse.rows; i++)
	{
		for (int j = 0; j < abstMat_3f_coarse.cols; j++)
		{
			float w = (float)face_mask.at<uchar>(i, j) / 255.0;
			abstMat.at<cv::Vec3f>(i, j) = abstMat_3f_fine.at<cv::Vec3f>(i, j)*w + abstMat_3f_coarse.at<cv::Vec3f>(i, j)*(1.0 - w);
		}
	}
	
	abstMat_4f.create(src_color.rows, src_color.cols, CV_32FC4);

	for (int i = 0; i < abstMat.rows; i++)
	{
		for (int j = 0; j < abstMat.cols; j++)
		{
			tempa = abstMat.at<cv::Vec3f>(i, j);
			abstMat_4f.at<cv::Vec4f>(i, j) = cv::Vec4f(tempa[0], tempa[1], tempa[2], 1.0);
		}
	}

	wet_level(ui.spinBox_wet->value());
		
}


/***********************************************************************
*	湿画法参数调节响应函数
***********************************************************************/
void WQPortrait::wet_level(int level)
{	
	
	if (src.empty())
	{
		QMessageBox::critical(0, "critical message", "Please open an image!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	if (face_mask.empty())
	{
		QMessageBox::critical(0, "critical message", "mask is empty!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	
	wetfilter.create(abstMat.rows, abstMat.cols, CV_32FC4);
	if (flag == 0){
		wet_main_cuda(abstMat, wetfilter, randomNoiseImage, level, face_mask, flag);

	}
	else{
		wet_main_cuda(abstMat, wetfilter, randomNoiseImage, level, error_mask, flag);

	}
	/*if (is_video)
		addWeighted(abstMat_4f, 0.6, wetfilter,0.4,0, wetfilter);*/
	

	darken_level(ui.doubleSpinBox_darken->value());
}

/***********************************************************************
*	边缘加深参数调节响应函数
***********************************************************************/
void WQPortrait::darken_level(double level)
{
	if (src.empty())
	{
		QMessageBox::critical(0, "critical message", "Please open an image!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	if (face_mask.empty())
	{
		QMessageBox::critical(0, "critical message", "mask is empty!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	StyleTransfer *st = new StyleTransfer();
	if (flag == 0)
	    darken = st->edgeDarken(wetfilter, abstMat_4f, face_mask, level, flag);//output: CV_32FC4
	else
		darken = st->edgeDarken(wetfilter, abstMat_4f, error_mask, level, flag);//output: CV_32FC4
	
	delete(st);
	st = nullptr;
	
	
	if (ui.checkBox_white->isChecked())
		is_white(2);
	else
		is_white(0);
		
}


/***********************************************************************
*	是否留白响应函数
*   2为留白，0为不留白
***********************************************************************/
void WQPortrait::is_white(int index)
{
	if (src.empty())
	{
		QMessageBox::critical(0, "critical message", "Please open an image!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}		
	if (face_mask.empty())
	{
		QMessageBox::critical(0, "critical message", "mask is empty!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	StyleTransfer *st = new StyleTransfer();
	
	if (flag == 0)
	    boundDistort = st->boundaryDistort(darken, face_mask, perlinNoiseImage, index, flag);//output: CV_32FC4，input: CV_32FC4
	else
		boundDistort = st->boundaryDistort(darken, error_mask, perlinNoiseImage, index, flag);
	delete(st);
	st = nullptr;
	
	pigment_level(ui.doubleSpinBox_pigment->value());
}


/***********************************************************************
*	粗糙紊流参数调节响应函数
***********************************************************************/
void WQPortrait::pigment_level(double level)
{
	if (src.empty())
	{
		QMessageBox::critical(0, "critical message", "Please open an image!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	if (face_mask.empty())
	{
		QMessageBox::critical(0, "critical message", "mask is empty!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	/*if (is_video)
	{
		level -= 0.3;
	}*/
	StyleTransfer *st = new StyleTransfer();
	if (flag == 0)
	    pigment = st->pigmentVariation(boundDistort, randomNoiseImage, perlinNoiseImage, paper, face_mask, level, flag);//output: CV_32FC4
	else
		pigment = st->pigmentVariation(boundDistort, randomNoiseImage, perlinNoiseImage, paper, error_mask, level, flag);
	delete(st);
	st = nullptr;

	if (ui.dry_brush->isChecked())
		is_dry(2);
	else
		is_dry(0);
}

//枯笔技法
void WQPortrait::is_dry(int state)
{
	if (src.empty())
	{
		QMessageBox::critical(0, "critical message", "Please open an image!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	if (state == 2)//checked
	{
		dry_brush();
	}
	//if (!is_video)
	show_result(ui.comboBox_show->currentIndex());
}

//显示结果
void WQPortrait::show_result(int index)
{
	if (src.empty())
	{
		QMessageBox::critical(0, "critical message", "Please open an image!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	cv::Mat show_mat;
	if (index == 0)
	{
		if (ui.dry_brush->isChecked())
			show_mat = pigment_dry;
		else
			show_mat = pigment;
	}
	/*else if (index == 1)
	{
		show_mat = src;
	}*/
	else if (index == 1)
	{
		show_mat = abstMat_4f;
		//show_mat = src_color;//frd
	}
	else if (index == 2)
	{
		show_mat = wetfilter;
	}
	else if (index == 3)
	{
		show_mat = darken;
	}
	else
	{
		show_mat = boundDistort;
	}

	Tool *tool = new Tool();
	QImage result = tool->cvMat2QImage(show_mat);
	delete(tool);
	tool = nullptr;
	label_result->setPixmap(QPixmap::fromImage(result));
}

//cv::Mat readflo(cv::String addr)
//{
//	std::ifstream fin(addr, std::ios::binary);
//	cv::Mat error;
//	char buffer[sizeof(float)];
//	fin.read(buffer, sizeof(float));
//	float tar = ((float*)buffer)[0];
//	if (tar - 202021.25>1e-7) {
//		fin.close();
//		return error;
//	}
//	fin.read(buffer, sizeof(int));
//	int high = ((int*)buffer)[0];
//	fin.read(buffer, sizeof(int));
//	int width = ((int*)buffer)[0];
//	
//	cv::Mat flo = cv::Mat(cv::Size(high, width), CV_32FC2);
//	for (int i = 0; i < width; i++) {
//		for (int j = 0; j < high; j++) {
//			if (!fin.eof()) {
//				float * data = flo.ptr<float>(i, j);
//				fin.read(buffer, sizeof(float));
//				data[0] = ((float*)buffer)[0];
//				fin.read(buffer, sizeof(float));
//				data[1] = ((float*)buffer)[0];
//			}
//		}
//	}
//	fin.close();
//	return flo;
//}
//
////warp a image from optical flow
//cv::Mat warp_image(cv::Mat img, cv::Mat flow)
//{
//	cv::Mat map(flow.size(), CV_32FC2);
//	for (int y = 0; y < map.rows; y++)
//	{
//		for (int x = 0; x < map.cols; x++)
//		{
//			cv::Point2f f = -flow.at<cv::Point2f>(y, x);//pay attention the -
//			map.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
//		}
//	}
//	std::vector<cv::Mat> mats;
//	cv::split(map, mats);
//	cv::Mat warp_img;
//	remap(img, warp_img, mats[0], mats[1], CV_INTER_LINEAR);
//	return warp_img;
//}
//void read_all_flows(cv::String path, int size, std::vector<cv::Mat> &flow_forward_vector, std::vector<cv::Mat> &flow_backward_vector)
//{
//	for (int i = 0; i < size; i++)
//	{
//		char ss[10];
//		sprintf(ss, "%05d", i);
//		flow_forward_vector.push_back(readflo(path + "flowf_" + ss + ".flo"));
//		flow_backward_vector.push_back(readflo(path + "flowb_" + ss + ".flo"));
//	}
//}
//void WQPortrait::forward_backward_watercolor(std::vector<cv::Mat> frames, std::vector<int> keyframe_index, std::vector<cv::Mat>& watercolor_frames)
//{
//	//计算关键帧watercolor结果
//	//std::vector<Mat> watercolor_keyframes;
//	cv::String path = "H:\\Users\\duoduo\\Desktop\\flamingo_out\\";
//	face_mask.create(frames[0].rows, frames[0].cols, CV_8UC1);
//	flag = 0;// image mode
//	for (int i = 0; i < keyframe_index.size(); i++)
//	{
//		frames[keyframe_index[i]].copyTo(src);
//		abstraction_level(ui.spinBox_abst->value());
//		cv::Mat output;
//		cvtColor(pigment, output, CV_RGBA2RGB);
//		output.convertTo(output, CV_8UC3, 255);
//		//watercolor_keyframes.push_back(output);
//		output.copyTo(watercolor_frames[keyframe_index[i]]);
//	}
//	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(23, 23));//kernel for dilate
//	//std::vector<Mat> forward_warped_img(frames.size());//CV_8UC3
//
//	for (int m = 0; m < keyframe_index.size()-1; m++)
//	{
//		//从后向前		
//		for (int n = keyframe_index[m+1]; n > keyframe_index[m]; n--)
//		{		
//			//watercolor_keyframes[m + 1].copyTo(forward_warped_img[keyframe_index[m + 1]]);
//			char ss[10];
//			sprintf(ss, "%05d", n-1);
//			char s2[10];
//			sprintf(s2, "%05d", n);
//			cv::Mat flow_forward = readflo(path + "flowf_" + ss + ".flo");
//			cv::Mat warped_image = warp_image(watercolor_frames[n], flow_forward);//first warp result
//			//imwrite(path2 + "warped_image" + s2 + ".jpg", warped_image);
//			cv::Mat error_mask_src = imread(path + "maskforward_" + s2 + ".jpg", cv::IMREAD_GRAYSCALE);
//			cv::dilate(error_mask_src, error_mask, element);
//			flag = 1;//video mode
//			frames[n-1].copyTo(src);
//			abstraction_level(ui.spinBox_abst->value());
//			float weight;
//			cv::Mat pigment_and_warp(src.size(), CV_8UC3);
//
//			for (int i = 0; i < error_mask.rows; i++)
//			{
//				for (int j = 0; j < error_mask.cols; j++)
//				{
//					if (error_mask.at<uchar>(i, j) > 200)
//						weight = 1.0;
//					else
//						weight = 0.0;
//					cv::Vec4f p = pigment.at<cv::Vec4f>(i, j)*weight;
//					cv::Vec3b w = warped_image.at<cv::Vec3b>(i, j);
//					cv::Vec3b o;
//					o[0] = (p[0] + w[0] / 255.0*(1 - weight)) * 255;
//					o[1] = (p[1] + w[1] / 255.0*(1 - weight)) * 255;
//					o[2] = (p[2] + w[2] / 255.0*(1 - weight)) * 255;
//					pigment_and_warp.at<cv::Vec3b>(i, j) = o;
//				}
//			}
//			if (watercolor_frames[n - 1].empty())
//				pigment_and_warp.copyTo(watercolor_frames[n - 1]);
//		}
//
//		//从前往后
//		cv::Mat preframe;
//		watercolor_frames[keyframe_index[m]].copyTo(preframe);
//		//watercolor_keyframes[m].copyTo(preframe);
//		for (int k = keyframe_index[m]+1; k <keyframe_index[m+1]; k++)
//		{
//			char ss[10];
//			sprintf(ss, "%05d", k - 1);
//			cv::Mat flow_backward = readflo(path + "flowb_" + ss + ".flo");
//			cv::Mat warped_image = warp_image(preframe, flow_backward);//first warp result
//
//			cv::Mat error_mask_src = imread(path + "maskback_" + ss + ".jpg", cv::IMREAD_GRAYSCALE);
//			dilate(error_mask_src, error_mask, element);
//			flag = 1;//video mode
//			frames[k].copyTo(src);
//			abstraction_level(ui.spinBox_abst->value());
//			float weight;
//			cv::Mat pigment_and_warp(src.size(), CV_8UC3);
//
//			for (int i = 0; i < error_mask.rows; i++)
//			{
//				for (int j = 0; j < error_mask.cols; j++)
//				{
//					if (error_mask_src.at<uchar>(i, j) > 200)
//						weight = 1.0;//pigment's weight
//					else if (error_mask.at<uchar>(i, j) > 200)
//						weight = 0.5;
//					else
//						weight = 0.0;
//					cv::Vec4f p = pigment.at<cv::Vec4f>(i, j)*weight;
//					cv::Vec3b w = warped_image.at<cv::Vec3b>(i, j);
//					cv::Vec3b o;
//					o[0] = (p[0] + w[0] / 255.0*(1 - weight)) * 255;
//					o[1] = (p[1] + w[1] / 255.0*(1 - weight)) * 255;
//					o[2] = (p[2] + w[2] / 255.0*(1 - weight)) * 255;
//					pigment_and_warp.at<cv::Vec3b>(i, j) = o;
//				}
//			}
//			pigment_and_warp.copyTo(preframe);
//
//			float weight_forward = (float)(k - keyframe_index[m]) / (keyframe_index[m + 1] - keyframe_index[m]);
//			cv::Mat forward_and_backward;
//			addWeighted(watercolor_frames[k], weight_forward, pigment_and_warp, 1.0 - weight_forward, 0.0, forward_and_backward);
//			/*if (k == 23)
//				imwrite("H:\\Users\\duoduo\\Desktop\\f_out\\pp23.jpg",forward_and_backward);*/
//			forward_and_backward.copyTo(watercolor_frames[k]);
//			//pigment_and_warp.copyTo(watercolor_frames[k]);
//		}
//	}
//}


//枯笔
void WQPortrait::dry_brush()
{
	int convolution_width = 9;//卷积核的半径
	cv::Mat line_edge;
	cv::Mat line = cv::imread(flow_path, cv::IMREAD_GRAYSCALE);//生成的曲线图
	if (line.empty())
	{
		src.copyTo(line);
		cvtColor(line, line, CV_BGR2GRAY);
	}
	
	blur(line, line, cv::Size(5, 5));//先用均值滤波器进行平滑去噪
	Canny(line, line_edge, 50, 150, 3);
	
	line_edge.convertTo(line_edge, CV_32FC1, 1 / 255.0);
	
	cv::Mat line2;//压力图
	line_edge.copyTo(line2);
	for (int i = 0; i < line_edge.rows; i++)
	{
		for (int j = 0; j < line_edge.cols; j++)
		{
			if (i - 9 < 0 || j - 9 < 0 || i + 9 >= line_edge.rows || j + 9 >= line_edge.cols)
			{
				line2.at<float>(i, j) = 0.0;
				continue;
			}
			else
			{
				float result = (line_edge.at<float>(i - 9, j) + line_edge.at<float>(i - 5, j) + line_edge.at<float>(i - 4, j) + line_edge.at<float>(i, j) +
					line_edge.at<float>(i + 9, j) + line_edge.at<float>(i + 5, j) + line_edge.at<float>(i + 4, j) +
					line_edge.at<float>(i, j + 9) + line_edge.at<float>(i, j + 5) + line_edge.at<float>(i, j + 4) +
					line_edge.at<float>(i, j - 9) + line_edge.at<float>(i, j - 5) + line_edge.at<float>(i, j - 4) +
					line_edge.at<float>(i - 4, j - 4) + line_edge.at<float>(i - 4, j + 4) + line_edge.at<float>(i + 4, j + 4) + line_edge.at<float>(i + 4, j - 4) +
					line_edge.at<float>(i - 6, j - 6) + line_edge.at<float>(i - 6, j + 6) + line_edge.at<float>(i + 6, j + 6) + line_edge.at<float>(i + 6, j - 6) +
					line_edge.at<float>(i - 7, j - 6) + line_edge.at<float>(i - 7, j + 6) + line_edge.at<float>(i + 7, j + 6) + line_edge.at<float>(i + 7, j - 6) +
					line_edge.at<float>(i - 6, j - 7) + line_edge.at<float>(i - 6, j + 7) + line_edge.at<float>(i + 6, j + 7) + line_edge.at<float>(i + 6, j - 7));
				result = result / 28.0;
				line2.at<float>(i, j) = result;

			}
		}
	}
	
	//读取高度图
	cv::Mat paper_height = cv::imread("data/paperheight6.png", cv::IMREAD_GRAYSCALE);
	paper_height.convertTo(paper_height, CV_32FC1, 1 / 255.0);
	//float threshold_value = 0.01;//空白区域所占的百分比
	cv::Mat cal_sum(line2.size(), CV_8UC1);

	for (int i = 0; i < pigment.rows; i++)
	{
		for (int j = 0; j < pigment.cols; j++)
		{
			int ih = (int)((float)(i)* paper_height.rows / ((float)(pigment.rows)));
			int jh = (int)((float)(j)* paper_height.cols / ((float)(pigment.cols)));
			cal_sum.at<uchar>(i, j) = MIN((uchar)((paper_height.at<float>(ih, jh) * 2.0 + line2.at<float>(i, j)* 5.0) / 7.0*255.0), 255);
		}
	}
	
	pigment.copyTo(pigment_dry);
	for (int i = 0; i < cal_sum.rows; i++)
	{
		for (int j = 0; j < cal_sum.cols; j++)
		{
			if (cal_sum.at<uchar>(i, j) >50 && face_mask.at<uchar>(i, j)<100)//面部不添加
			{
				pigment_dry.at<cv::Vec4f>(i, j) = cv::Vec4f(1.0f, 1.0f, 1.0f, 1.0f);
			}
			/*if (cal_sum.at<uchar>(i, j) > threshold_pixel_value && mask.at<uchar>(i, j)>10 && mask.at<uchar>(i, j) < 240)
			{
			dry_brush.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}*/
		}
	}

}


//亮度饱和度调整
void WQPortrait::adjustColor(cv::Mat frame, cv::Mat& output)
{
	cv::Mat  image, hsvimage;
	const int max_Increment = 200;
	int Increment_value = 150;
	float Increment = (Increment_value - 100)* 1.0 / max_Increment;
	image = frame.clone();
	for (int i = 0; i < image.cols; ++i)
	{
		for (int j = 0; j < image.rows; ++j)
		{
			uchar b = image.at<cv::Vec3b>(j, i).val[0];
			uchar g = image.at<cv::Vec3b>(j, i).val[1];
			uchar r = image.at<cv::Vec3b>(j, i).val[2];
			float max = MAX(MAX(r, g), b);
			float min = MIN(MIN(r, g), b);
			float delta, value;
			uchar R_new, G_new, B_new;
			float L, S, alpha;
			delta = (max - min) / 255;
			if (delta == 0)
				continue;
			value = (max + min) / 255;
			L = value / 2;
			if (L < 0.5)
				S = delta / value;
			else
				S = delta / (2 - value);
			if (Increment >= 0)
			{
				if ((Increment + S) >= 1)
					alpha = S;
				else
					alpha = 1 - Increment;
				alpha = 1 / alpha - 1;
				R_new = r + (r - L * 255) * alpha;

				G_new = g + (g - L * 255) * alpha;

				B_new = b + (b - L * 255) * alpha;
			}
			else
			{
				alpha = Increment;
				R_new = L * 255 + (r - L * 255) * (1 + alpha);

				G_new = L * 255 + (g - L * 255) * (1 + alpha);

				B_new = L * 255 + (b - L * 255) * (1 + alpha);
			}
			image.at<cv::Vec3b>(j, i).val[0] = B_new;
			image.at<cv::Vec3b>(j, i).val[1] = G_new;
			image.at<cv::Vec3b>(j, i).val[2] = R_new;
		}
	}
	
	cvtColor(image, hsvimage, CV_BGR2HLS);
	for (int i = 0; i < hsvimage.cols; ++i)
	{
		for (int j = 0; j < hsvimage.rows; ++j)
		{
			hsvimage.at<cv::Vec3b>(j, i).val[1] = cv::saturate_cast<uchar>(hsvimage.at<cv::Vec3b>(j, i).val[1] + 5);//亮度
			//hsvimage.at<Vec3b>(j, i).val[2] = saturate_cast<uchar>(hsvimage.at<Vec3b>(j, i).val[1] + 5);//饱和度

		}
	}
	cvtColor(hsvimage, image, CV_HLS2BGR);
	image.copyTo(output);
}


//视频处理按钮响应函数
void WQPortrait::video_process()
{
	if (first_frame.empty())
	{
		QMessageBox::critical(0, "critical message", "Please open a video!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	is_video = true;
	//is_video = false;
	
	std::string video_file_name = file_name;
	int pos2 = video_file_name.find(".");
	video_file_name.replace(pos2, 1, "_watercolor.");
	cv::VideoWriter writer(video_file_name, CV_FOURCC('D', 'I', 'V', 'X'), cap.get(CV_CAP_PROP_FPS),
		cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
	if (!writer.isOpened())
	{
		QMessageBox::critical(0, "critical message", "can't write video", QMessageBox::Ok | QMessageBox::Escape, 0);

		return;
	}
	cv::Mat f, output;
	first_frame.copyTo(f);
	do
	{
		f.copyTo(src);
		saliency(src);
		adjustColor(src, src_color);//调整颜色
		abstraction_level(ui.spinBox_abst->value());
		if (ui.dry_brush->isChecked())
			cvtColor(pigment_dry, output, CV_RGBA2RGB);
		else
			cvtColor(pigment, output, CV_RGBA2RGB);
		output.convertTo(output, CV_8UC3, 255);
		writer << output;
	} while (cap.read(f));
	QMessageBox::information(0, QString::fromLocal8Bit("提示"), QString::fromLocal8Bit("视频转换完成！"), QMessageBox::Ok | QMessageBox::Escape, 0);
}


//图像处理按钮响应函数
void WQPortrait::image_process()
{
	if (img->isNull())
	{
		QMessageBox::critical(0, "critical message", "Please open a file!", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	//处理路径
	int pos;
	int pos_copy;
	pos = file_name.find("/");
	while (pos > -1)
	{
		file_name.replace(pos, 1, "\\");
		pos_copy = pos;
		pos = file_name.find("/");
	}
	int p = file_name.find(".");
	std::string suffix = file_name.substr(p);//获取后缀

	//图像处理
	if (suffix == ".jpg" || suffix == ".png" || suffix == ".bmp" || suffix == ".JPG" || suffix == ".PNG" || suffix == ".BMP"){
		std::string image_name = file_name.substr(pos_copy + 1);
		std::string mask_path = "H:\\Users\\duoduo\\Documents\\Visual Studio 2013\\Projects\\WQPortrait\\WQPortrait\\image\\dry\\";
		flow_path = file_name.substr(pos_copy + 1);
		pos = image_name.find(".");
		image_name.replace(pos, 1, "_mask.");
		mask_path += image_name;
		flow_path.replace(pos, 1, "_out.");
		flow_path = "H:\\Users\\duoduo\\Documents\\Visual Studio 2013\\Projects\\WQPortrait\\WQPortrait\\image\\dry\\" + flow_path;

		src = cv::imread(file_name);

		/*****************************人脸检测/显著性物体检测********************************/
		//SkinDetect *sd = new SkinDetect(src);
		//face_mask.create(src.rows, src.cols, CV_8UC1);
		//bool is_face = sd->face_detect(face_mask);//是否检测到人脸
		//bool is_face = false;
		
		//face_mask = imread(mask_path, cv::IMREAD_GRAYSCALE);
		//if (face_mask.empty())
			saliency(src);

		adjustColor(src, src_color);//调整颜色
		abstraction_level(ui.spinBox_abst->value());


	}//image
	else//video
	{
		first_frame.copyTo(src);
		saliency(src);
		adjustColor(src, src_color);//调整颜色
		abstraction_level(ui.spinBox_abst->value());
	}
}


void WQPortrait::test()
{	
	/*string image_path="image/ball2.jpg";
	src = imread(image_path);
	face_mask.create(src.rows, src.cols, CV_8UC1);
	cv::Mat abstMat = cv::imread(image_path);*/	
	/*abstMat.convertTo(abstMat, CV_32FC3, 1 / 255.0);
	Mat face_mask(abstMat.size(), CV_8UC1, Scalar::all(0));
	Mat abstMat2(abstMat.rows-30, abstMat.cols-30, abstMat.type());
	for (int i = 30; i < abstMat.rows; i++)
	{
		for (int j = 30; j < abstMat.cols; j++)
		{
			abstMat2.at<Vec3f>(i - 30, j - 30) = abstMat.at<Vec3f>(i, j);
		}
	}*/
	/*Mat output2;
	abstraction_level(ui.spinBox_abst->value());
	cvtColor(pigment, output2, CV_RGBA2RGB);
	output2.convertTo(output2, CV_8UC3, 255);
	imwrite("video/ball2_result.jpg",output2);
	return;*/
	//cv::Mat abstMat2 = cv::imread("image/ball3.jpg");
	//abstMat2.convertTo(abstMat2, CV_32FC3, 1 / 255.0);
	
	//Mat abst;

	//long time1 = clock();
	//cv::ximgproc::fastGlobalSmootherFilter(src2, src2, abst, 600, 3, 0.25, 3);
	////cv::ximgproc::guidedFilter(src2, src2, abst, 16.0, 1300.5, -1);
	//imwrite("image/fast.jpg", abst);
	//long time2 = clock();
	//QMessageBox::critical(0, "critical message", QString::number(time2-time1), QMessageBox::Ok | QMessageBox::Escape, 0);

	/*******************************抽象简化***************************************/
	
	
	//StyleTransfer *st = new StyleTransfer();

	//cv::Mat nearColor = st->nearColorEdgeDetector(abstMat); //32FC4,only 3 type: 1 -1 0	
	//cv::Mat nearColor2 = st->nearColorEdgeDetector(abstMat2); //32FC4,only 3 type: 1 -1 0	
	//cv::Mat largeGradients = st->largeGradient(abstMat); //32FC3 u v 0,检测x和y方向的梯度
	//cv::Mat largeGradients2 = st->largeGradient(abstMat2); //32FC3 u v 0,检测x和y方向的梯度
	//cv::Mat edgeType = st->edgeTypeDetect(abstMat, nearColor, largeGradients, face_mask);
	//cv::Mat edgeType2 = st->edgeTypeDetect(abstMat2, nearColor2, largeGradients2, face_mask);
	//cv::Mat wetpre = st->wetPre(abstMat, edgeType, largeGradients, randomNoiseImage);
	//cv::Mat wetpre2 = st->wetPre(abstMat2, edgeType2, largeGradients2, randomNoiseImage);
	//edgeType.convertTo(edgeType, CV_8UC4, 255);
	//imwrite("image/bedgeType.png", edgeType);
	//edgeType2.convertTo(edgeType2, CV_8UC4, 255);
	//imwrite("image/bedgeType2.png", edgeType2);
	//cv::Mat wetfilter = st->wetFilter(wetpre, largeGradients);//output: CV_32FC4

	//imshow("nearcolor", nearColor);
	//imshow("nearcolor2", nearColor2);
	//imshow("largeGradients", largeGradients);
	//imshow("largeGradients2", largeGradients2);
	//imshow("edgeType", edgeType);
	//imshow("edgeType2", edgeType2);
	/*imshow("wetpre", wetpre);
	imshow("wetpre2", wetpre2);*/

	/*Mat output;
	cvtColor(wetfilter, output, CV_RGBA2RGB);
	output.convertTo(output, CV_8UC3, 255);
	imwrite("image/ball1_wet_CPU.jpg", output);*/
	//Mat wetfilter(abstMat.rows, abstMat.cols, CV_32FC4);
	//wet_main_cuda(abstMat, wetfilter, randomNoiseImage);
	
	//cv::Mat darken = st->edgeDarken(wetfilter, abstMat_4f, face_mask);//output: CV_32FC4
		
	//cv::Mat boundDistort = st->boundaryDistort(darken, face_mask, perlinNoiseImage);//output: CV_32FC4，input: CV_32FC4
	
	//Mat texture = imread("data/texture.png");
	//4.pigment variation
	//cv::Mat pigment = st->pigmentVariation(boundDistort, randomNoiseImage, texture, paper, face_mask);//output: CV_32FC4

	//pigment.convertTo(pigment, CV_8UC3, 255);	
	//imwrite("image/33_final.jpg",pigment);

	//delete(st);
	//st = nullptr;

	//test
	//Mat element2 = getStructuringElement(MORPH_RECT, Size(15, 15));//kernel for dilate
	//src = imread("H:\\Users\\duoduo\\Desktop\\00078.jpg");	
	//face_mask.create(src.rows, src.cols, CV_8UC1);
	//error_mask = imread("H:\\Users\\duoduo\\Desktop\\flamingo_out\\maskforward_00079.jpg", IMREAD_GRAYSCALE);
	//	
	//dilate(error_mask, error_mask, element2);
	//flag = 1;
	//abstraction_level(ui.spinBox_abst->value());
	//Mat output2;
	//cvtColor(pigment, output2, CV_RGBA2RGB);
	//output2.convertTo(output2, CV_8UC3, 255);
	//imwrite("image/output.jpg",output2);
	///*for (int i = 0; i < src.rows; i++)
	//{
	//	for (int j = 0; j < src.cols; j++)
	//	{
	//		if (error_mask.at<uchar>(i, j) < 200)
	//			src.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
	//	}
	//}
	//imwrite("image/output2.jpg", src);*/
	//return;
//Mat im1 = imread("H:\\Users\\duoduo\\Desktop\\00008.jpg");
////Mat im2 = imread("H:\\Users\\duoduo\\Desktop\\00008.jpg");
//Mat flow_forward2 = readflo("H:\\Users\\duoduo\\Desktop\\flamingo_out\\flowb_00007.flo");
//long time1 = clock();
//Mat warped_image = warp_image(im1, flow_forward2);//first warp result
//long time2 = clock();
//QMessageBox::critical(0, "critical message", QString::number(time2 - time1), QMessageBox::Ok | QMessageBox::Escape, 0);
////imwrite("image/warped_image.jpg",warped_image);
//return;

	//视频处理******************************
	//使用光流处理
	//VideoCapture capture;
	//Mat frame, output;
	//capture.open("video/flamingo_input.mp4");
	//if (!capture.isOpened())
	//{
	//	QMessageBox::critical(0, "critical message", "can't open", QMessageBox::Ok | QMessageBox::Escape, 0);
	//	return;
	//}
	//
	//VideoWriter writer("video/flamingo_keyframe23.mp4", CV_FOURCC('D', 'I', 'V', 'X'), capture.get(CV_CAP_PROP_FPS),
	//	Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
	//if (!writer.isOpened())
	//{
	//	QMessageBox::critical(0, "critical message", "can't write", QMessageBox::Ok | QMessageBox::Escape, 0);

	//	return;
	//}

	//int count = 0;
	//int frame_rate = capture.get(CV_CAP_PROP_FPS);
	//std::vector<Mat> frames;
	////std::vector<Mat> forward_watercolor;
	//std::vector<int> keyframe_index;
	//while (capture.read(frame))
	//{		
	//	Mat frame_temp;
	//	frame.copyTo(frame_temp);
	//	frames.push_back(frame_temp);//注意读取可能出现错误，只保留最后一帧
	//	if (count%frame_rate == 0)
	//		keyframe_index.push_back(count);
	//	//char ss[10];
	//	//sprintf(ss, "%05d", count);
	//	//randomNoiseImage = cv::imread(path+"random_"+ss+".png");
	//	//perlinNoiseImage = cv::imread(path + "perlin_" + ss + ".png");
	//	//frame.copyTo(src);		
	//	//face_mask.create(frame.rows, frame.cols, CV_8UC1);
	//	////abstraction_level(ui.spinBox_abst->value());
	//	//abstraction_level(6);
	//	//cvtColor(pigment, output, CV_RGBA2RGB);
	//	//output.convertTo(output, CV_8UC3, 255);
	//	//writer << frame;
	//	count++;		
	//}
	//int video_length = frames.size();
	//if (keyframe_index[keyframe_index.size() - 1] != video_length-1)
	//{
	//	keyframe_index.push_back(video_length - 1);
	//}
	//int keyframe_num = keyframe_index.size();
	////String flow_path = "H:\\Users\\duoduo\\Desktop\\flamingo_out\\";
	//std::vector<Mat> watercolor_frames(video_length);
	//forward_backward_watercolor(frames, keyframe_index, watercolor_frames);
	//String p = "H:\\Users\\duoduo\\Desktop\\f_out\\";
	//for (int i = 0; i < video_length; i++)
	//{
	//	writer << watercolor_frames[i];
	//	/*char ss[10];
	//	sprintf(ss, "%05d", i);
	//	
	//	imwrite( p+ "aapigment"+ss + ".jpg", watercolor_frames[i]);*/
	//}

	//capture.release();
	//writer.release();

//视频处理结束*****************************************


//从文件夹中读取图片并逐帧处理***************************
	//String img_path = "H:\\Users\\duoduo\\Desktop\\eccv18\\train\\input\\duoduo\\";
	//String filename = "tuk-tuk\\";
	//std::vector<String> img;
	//glob(img_path+filename, img, false);
	//size_t count = img.size();
	//Mat output;
	//flag = 0;

	//for (size_t i = 0; i < count; i++)
	//{
	//	/*stringstream str;
	//	str << i << ".jpg";*/
	//	char ss[10];
	//	sprintf(ss, "%05d", i);
	//	Mat frame = imread(img_path + filename + ss + ".jpg");

	//	//Mat frame = imread("image\\00000.jpg");
	//	Mat output;
	//	if (!frame.empty())
	//	{		
	//		frame.copyTo(src);		
	//		//saliency(src);
	//		face_mask.create(frame.rows, frame.cols, CV_8UC1);
	//		//abstraction_level(ui.spinBox_abst->value());
	//		abstraction_level(4);
	//		cvtColor(pigment, output, CV_RGBA2RGB);
	//		output.convertTo(output, CV_8UC4, 255);
	//		String output_path = "H:\\Users\\duoduo\\Desktop\\eccv18\\train\\processed\\watercolor\\duoduo\\";
	//		imwrite(output_path + filename +ss + ".jpg", output);
	//
	//	}
	//	
	//}
//从文件夹中读取图片并逐帧处理结束***************************
	
//单一图片渲染
	//Mat a = imread("image/dry/20.jpg");
	//a.copyTo(src);
	////saliency(src);
	////Mat ff(src.rows, src.cols, CV_8UC1, Scalar::all(0));
	//Mat ff = imread("image/dry/20_mask.jpg", IMREAD_GRAYSCALE);
	//ff.copyTo(face_mask);
	//flag = 0;
	//Mat outp;
	//
	//abstraction_level(4);
	///*cvtColor(abstMat_4f, outp, CV_RGBA2RGB);
	//outp.convertTo(outp, CV_8UC4, 255);
	//imwrite("image/dry/21_abst2.jpg", outp);

	//cvtColor(wetfilter, outp, CV_RGBA2RGB);
	//outp.convertTo(outp, CV_8UC4, 255);
	//imwrite("image/dry/21_wet2.jpg", outp);

	//cvtColor(darken, outp, CV_RGBA2RGB);
	//outp.convertTo(outp, CV_8UC4, 255);
	//imwrite("image/dry/21_darken2.jpg", outp);

	//cvtColor(boundDistort, outp, CV_RGBA2RGB);
	//outp.convertTo(outp, CV_8UC4, 255);
	//imwrite("image/dry/21_bd2.jpg",outp);*/
	////imshow("wet", boundDistort);
	//cvtColor(pigment, outp, CV_RGBA2RGB);
	//outp.convertTo(outp, CV_8UC4, 255);
	//imwrite("image/dry/20_pigment3.jpg", outp);
	////long time2 = clock();
	////QMessageBox::critical(0, "critical message", QString::number(time2 - time1), QMessageBox::Ok | QMessageBox::Escape, 0);
	//return;
	//单一图片渲染结束

	cv::VideoCapture capture;
	cv::Mat frame, output;
	flag = 0;
	capture.open("H:\\Users\\duoduo\\Desktop\\portrait_video\\29.mp4");
	if (!capture.isOpened())
	{
		QMessageBox::critical(0, "critical message", "can't open", QMessageBox::Ok | QMessageBox::Escape, 0);
		return;
	}
	
	cv::VideoWriter writer("H:\\Users\\duoduo\\Desktop\\portrait_video\\29_watercolor_wet20.mp4", CV_FOURCC('D', 'I', 'V', 'X'), capture.get(CV_CAP_PROP_FPS),
		cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT)));
	if (!writer.isOpened())
	{
		QMessageBox::critical(0, "critical message", "can't write", QMessageBox::Ok | QMessageBox::Escape, 0);

		return;
	}
	while (capture.read(frame)){
		frame.copyTo(src);
		saliency(src);
		//face_mask.create(frame.rows, frame.cols, CV_8UC1);
		//abstraction_level(ui.spinBox_abst->value());
		abstraction_level(10);
		cvtColor(pigment, output, CV_RGBA2RGB);
		output.convertTo(output, CV_8UC3, 255);
		//String output_path = "H:\\Users\\duoduo\\Desktop\\eccv18\\train\\processed\\watercolor\\duoduo\\";
		//imwrite("H:\\Users\\duoduo\\Desktop\\portrait_video\\10_watercolor.jpg", output);
		//imwrite("H:\\Users\\duoduo\\Desktop\\portrait_video\\facemask.jpg", face_mask);
		//break;
		cv::Mat output2;
		adjustColor(output, output2);
		writer << output2;
	}
}
