#include "Tool.h"


Tool::Tool()
{
}


Tool::~Tool()
{
}

cv::Mat Tool::convolveVertical1D(cv::Mat f, float* kernel, int k_len)
{
	int r = (k_len - 1) / 2;

	cv::Mat out = cv::Mat(f.rows, f.cols, CV_32FC3);
	for (int y = 0; y < f.rows; y++) {
		for (int x = 0; x < f.cols; x++) {
			cv::Vec3f sum = cv::Vec3f(0.f, 0.f, 0.f);

			// center pixel
			cv::Vec3f center = f.at<cv::Vec3f>(y, x);
			sum += kernel[r] * center;
			

			// neighbor pixels
			for (int i = 1; i <= r; ++i)
			{
				cv::Vec3f a = f.at<cv::Vec3f>(cv::max(0, y - i), x);
				cv::Vec3f b = f.at<cv::Vec3f>(cv::min(f.rows - 1, y + i), x);

				sum += kernel[r + i] * b;
				
				sum += kernel[r - i] * a;
				
			}
			out.at<cv::Vec3f>(y, x) = sum;
		}
	}
	return out;
}
cv::Mat Tool::convolveHorizontal1D(cv::Mat f, float* kernel, int k_len)
{
	int r = (k_len - 1) / 2;

	cv::Mat out = cv::Mat(f.rows, f.cols, CV_32FC3);
	for (int y = 0; y < f.rows; y++) {
		for (int x = 0; x < f.cols; x++) {
			cv::Vec3f sum = cv::Vec3f(0.f, 0.f, 0.f);

			// center pixel
			cv::Vec3f center = f.at<cv::Vec3f>(y, x);
			sum += kernel[r] * center;

			// neighbor pixels
			for (int i = 1; i <= r; ++i)
			{
				cv::Vec3f a = f.at<cv::Vec3f>(y, cv::max(0, x - i));
				cv::Vec3f b = f.at<cv::Vec3f>(y, cv::min(x + i, f.cols - 1));

				sum += kernel[r + i] * b;

				sum += kernel[r - i] * a;
			}
			out.at<cv::Vec3f>(y, x) = sum;
		}
	}
	return out;
}
cv::Mat Tool::GaussianBlur(cv::Mat f, float sigma)
{
	int k_size = (int)(2.0f * floor(sqrt(-log(0.1f) * 2 * (sigma*sigma))) + 1.0f);
	int r = (k_size - 1) / 2;
	float sigmasquare = sigma * sigma;

	// compute gaussian kernel
	float *kernel = new float[k_size];
	for (int i = -r; i <= r; ++i)
	{
		kernel[i + r] = exp(-i*i / (2 * sigmasquare)) / (sqrt(sigmasquare* 6.2831855f));
	}

	// normalize kernel
	float sum = 0;
	for (int j = 0; j < k_size; ++j)
	{
		sum += kernel[j];
	}

	for (int j = 0; j < k_size; ++j)
	{
		kernel[j] = (kernel[j] / sum);
	}

	return convolveVertical1D(convolveHorizontal1D(f, kernel,k_size), kernel,k_size);
}

cv::Mat Tool::QImage2cvMat(QImage image)
{
	cv::Mat mat;
	//qDebug() << image.format();
	switch (image.format())
	{
	case QImage::Format_ARGB32:
	case QImage::Format_RGB32:
	case QImage::Format_ARGB32_Premultiplied:
		mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
		break;
	case QImage::Format_RGB888:
		mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
		//cv::cvtColor(mat, mat, CV_BGR2RGB);
		break;
	case QImage::Format_Indexed8:
		mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
		break;
	}
	return mat;
}

QImage Tool::cvMat2QImage(const cv::Mat& mat)
{
	// 8-bits unsigned, NO. OF CHANNELS = 1
	if (mat.type() == CV_8UC1)
	{
		QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
		// Set the color table (used to translate colour indexes to qRgb values)
		image.setColorCount(256);
		for (int i = 0; i < 256; i++)
		{
			image.setColor(i, qRgb(i, i, i));
		}
		// Copy input Mat
		uchar *pSrc = mat.data;
		for (int row = 0; row < mat.rows; row++)
		{
			uchar *pDest = image.scanLine(row);
			memcpy(pDest, pSrc, mat.cols);
			pSrc += mat.step;
		}
		return image;
	}
	// 8-bits unsigned, NO. OF CHANNELS = 3
	else if (mat.type() == CV_8UC3)
	{
		// Copy input Mat
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	else if (mat.type() == CV_8UC4)
	{
		qDebug() << "CV_8UC4";
		// Copy input Mat
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
		return image.copy();
	}
	else if (mat.type() == CV_32FC3 || mat.type() == CV_32FC4)
	{
		cv::Mat mat2, mat3;
		mat.convertTo(mat2,CV_8UC4,255);
		std::vector<cv::Mat> channels;
		cv::split(mat2, channels );
		channels.pop_back();
		cv::merge(channels, mat3);
		const uchar *pSrc = (const uchar*)mat3.data;
		// Create QImage with same dimensions as input Mat
		QImage image(pSrc, mat3.cols, mat3.rows, mat3.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	else
	{
		qDebug() << "ERROR: Mat could not be converted to QImage.";
		return QImage();
	}
}
