#include "SkinDetect.h"


SkinDetect::SkinDetect(cv::Mat src)
{
	cv::Mat resize_mat;
	is_resize = false;
	int resize_rows, resize_cols;
	if (src.rows > 256)
	{
		resize_rows = src.rows / 3;
		is_resize = true;
	}
	if (src.cols > 256)
	{
		resize_cols = src.cols / 3;
		is_resize = true;
	}
	rows = src.rows;
	cols = src.cols;
	cv::resize(src, src, cv::Size(resize_cols, resize_rows), (0, 0), (0, 0), cv::INTER_LINEAR);
	src.copyTo(img);
}


SkinDetect::~SkinDetect()
{
}

bool SkinDetect::face_detect(cv::Mat &face_mask)
{
	cv::Mat face = cv::Mat::zeros(img.size(), CV_8UC1);
	frontal_face_detector detector = get_frontal_face_detector();

	dlib::cv_image<rgb_pixel> img_2d(img);
	std::vector<rectangle> dets = detector(img_2d);

	if (dets.size() <= 0)
		return false;

	long padding=0;
	for (int i = 0; i < dets.size(); i++)
	{		
		long left = MIN(img.cols-1,MAX(0, dets[i].left() - padding));//扩展mask
		long right = MIN(img.cols - 1, MAX(0, dets[i].right() + padding));
		long top = MIN(img.rows - 1, MAX(0, dets[i].top() - padding));
		long bottom = MIN(img.rows - 1, MAX(0, dets[i].bottom() + padding));
		cv::rectangle(face, cvPoint(left, top), cvPoint(right, bottom), cv::Scalar(255, 255, 255), -1);//填充
		//cv::rectangle(img, cvPoint(left, top), cvPoint(right, bottom), cv::Scalar(255, 255, 255), 1);//填充
	}
	
	//shape_predictor sp;
	//std::vector<full_object_detection> shapes;
	//deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	//std::vector<std::vector<cv::Point >> contours;

	//for (int i = 0; i < dets.size(); i++)
	//{
	//	full_object_detection shape = sp(img_2d, dets[i]);
	//	std::vector<cv::Point > contour;
	//	//contour.reserve(27);
	//	for (int j = 0; j <= 16; j++)
	//	{
	//		contour.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));
	//	}
	//	for (int j = 26; j >= 17; j--)
	//	{
	//		contour.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));
	//	}
	//	contours.push_back(contour);
	//	
	//	//shapes.push_back(shape);
	//}
	//cv::polylines(face, contours, true, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);//第2个参数可以采用contour或者contours，均可
	//cv::fillPoly(face, contours, cv::Scalar(255, 255, 255));//fillPoly函数的第二个参数是二维数组！！
	
	cv::boxFilter(face, face, -1, cv::Size(21,21));
	if (is_resize)
	{
		cv::resize(face, face, cv::Size(cols, rows), (0, 0), (0, 0), cv::INTER_LINEAR);
	}
	face.copyTo(face_mask);
	return true;
}