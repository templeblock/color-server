#include "FaceDeformation.h"


FaceDeformation::FaceDeformation()
{
}


FaceDeformation::~FaceDeformation()
{
}




#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)

// 对矩阵求逆，结果保存在a中  
int f_GaussInverseMatMatrix(double A[], double B[], int nColumns)
{
	memcpy(B, A, sizeof(double) * nColumns * nColumns);
	int *is, *js, i, j, k, l, u, v;
	double d, p;
	is = new int[nColumns];
	js = new int[nColumns];
	for (k = 0; k <= nColumns - 1; k++)
	{
		d = 0.0;
		for (i = k; i <= nColumns - 1; i++)
			for (j = k; j <= nColumns - 1; j++)
			{
				l = i*nColumns + j; p = fabs(B[l]);
				if (p>d) { d = p; is[k] = i; js[k] = j; }
			}
		if (d + 1.0 == 1.0)
		{
			free(is); free(js); printf("err**not inv\n");
			return(1);
		}
		if (is[k] != k)
			for (j = 0; j <= nColumns - 1; j++)
			{
				u = k*nColumns + j; v = is[k] * nColumns + j;
				p = B[u]; B[u] = B[v]; B[v] = p;
			}
		if (js[k] != k)
			for (i = 0; i <= nColumns - 1; i++)
			{
				u = i*nColumns + k; v = i*nColumns + js[k];
				p = B[u]; B[u] = B[v]; B[v] = p;
			}
		l = k*nColumns + k;
		B[l] = 1.0f / B[l];
		for (j = 0; j <= nColumns - 1; j++)
			if (j != k)
			{
				u = k*nColumns + j; B[u] = B[u] * B[l];
			}
		for (i = 0; i <= nColumns - 1; i++)
			if (i != k)
				for (j = 0; j <= nColumns - 1; j++)
					if (j != k)
					{
						u = i*nColumns + j;
						B[u] -= B[i*nColumns + k] * B[k*nColumns + j];
					}
		for (i = 0; i <= nColumns - 1; i++)
			if (i != k)
			{
				u = i*nColumns + k;
				B[u] = -B[u] * B[l];
			}
	}
	for (k = nColumns - 1; k >= 0; k--)
	{
		if (js[k] != k)
			for (j = 0; j <= nColumns - 1; j++)
			{
				u = k*nColumns + j; v = js[k] * nColumns + j;
				p = B[u]; B[u] = B[v]; B[v] = p;
			}
		if (is[k] != k)
			for (i = 0; i <= nColumns - 1; i++)
			{
				u = i*nColumns + k; v = i*nColumns + is[k];
				p = B[u]; B[u] = B[v]; B[v] = p;
			}
	}
	free(is);
	free(js);
	return 0;
}


// 求矩阵的乘积 C = A.*B  
void f_MatMatrixMul(double a[], double b[], int m, int n, int k, double c[])
{
	int i, j, l, u;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < k; j++)
		{
			u = i * k + j;
			c[u] = 0.0;
			for (l = 0; l < n; l++)
			{
				c[u] += a[i * n + l] * b[l * k + j];
			}
		}
	}
}

//------------------------------------------------------------------    
//Function: MRLS 
//Params:   
//return: 0 or false    
//Reference: Nonrigid image deforcv::Mation using moving regularized least squares. 
//-------------------------------------------------------------------    
//unsigned char *srcData
cv::Mat f_MLSR(cv::Mat srcImg, std::vector<int> srcPoint, std::vector<int> dstPoint, int pointNum)
{
	int ret = 0;
	int width = srcImg.cols;
	int height = srcImg.rows;
	cv::Mat result(srcImg.rows, srcImg.cols, CV_8UC3);

	//unsigned char* tempData = (unsigned char*)malloc(sizeof(unsigned char)* height * stride);
	//memcpy(tempData, srcData, sizeof(unsigned char) * height * stride);
	//Process
	double *srcPx = (double*)malloc(sizeof(double) * pointNum);
	double *srcPy = (double*)malloc(sizeof(double) * pointNum);
	double *dstPx = (double*)malloc(sizeof(double) * pointNum);
	double *dstPy = (double*)malloc(sizeof(double) * pointNum);
	double* GammaIJ = (double*)malloc(sizeof(double) * pointNum * pointNum);
	double* GammaIJT = (double*)malloc(sizeof(double) * pointNum * pointNum);
	double* GammaP = (double*)malloc(sizeof(double) * pointNum);
	double* S = (double*)malloc(sizeof(double) * pointNum);
	double* W = (double*)malloc(sizeof(double) * pointNum * pointNum);
	double* WInv = (double*)malloc(sizeof(double) * pointNum * pointNum);
	double* tempIJ = (double*)malloc(sizeof(double) * pointNum * pointNum);
	double belta = 2.0;
	double alpha = 2.0;
	double lambda = 1.0;
	double fbelta = 1.0 / (belta * belta);
	for (int i = 0; i < pointNum; i++)
	{
		srcPx[i] = (double)srcPoint[2 * i] / width;
		srcPy[i] = (double)srcPoint[2 * i + 1] / height;
		dstPx[i] = (double)dstPoint[2 * i] / width;
		dstPy[i] = (double)dstPoint[2 * i + 1] / height;
	}
	int w = pointNum;
	for (int j = 0; j < pointNum; j++)
	{
		for (int i = 0; i < pointNum; i++)
		{
			GammaIJ[i + j * w] = exp(-((double)(srcPx[i] - srcPx[j]) * (srcPx[i] - srcPx[j]) + (srcPy[i] - srcPy[j]) * (srcPy[i] - srcPy[j])) * fbelta);
		}
	}
	//unsigned char* pSrc = srcData;
	int pos, pos1;
	cv::Vec3b r1, r2, r3, aa, bb, cc, dd;
	unsigned char *pSrcL1;
	unsigned char *pSrcL2;
	//unsigned char* p = tempData;
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			double px = (double)i / width, py = (double)j / height;
			for (int n = 0; n < pointNum; n++)
			{
				for (int m = 0; m < pointNum; m++)
				{
					if (n == m)
						W[m + n * w] = pow(((double)(px - srcPx[m]) * (px - srcPx[m]) + (py - srcPy[m]) * (py - srcPy[m])), -alpha);
					else
						W[m + n * w] = 0;
				}
			}
			//compute inverse cv::Matrix of W
			f_GaussInverseMatMatrix(W, WInv, pointNum);
			//compute Gamma + lambda * WInv
			for (int n = 0; n < pointNum; n++)
			{
				for (int m = 0; m < pointNum; m++)
				{
					//W is a diagonal cv::Matrix with the i-th entry Wi(p),other's 0
					GammaIJT[m + n * w] = GammaIJ[m + n * w] + lambda * WInv[m + n * w];
				}
			}
			//compute inverse cv::Matrix of (Gamma + lambda * WInv)
			f_GaussInverseMatMatrix(GammaIJT, tempIJ, pointNum);
			//compute GammaP   exp(-(p-xi) * (p - xi) / (belta * belta))
			for (int m = 0; m < pointNum; m++)
				GammaP[m] = exp((-((double)(px - srcPx[m]) * (px - srcPx[m]) + (py - srcPy[m]) * (py - srcPy[m])) * fbelta));
			//compute S
			f_MatMatrixMul(GammaP, tempIJ, 1, pointNum, pointNum, S);
			//compute fp   T + SY
			double sumx = 0, sumy = 0;
			for (int m = 0; m < pointNum; m++)
			{
				sumx += S[m] * srcPx[m];
				sumy += S[m] * srcPy[m];
			}
			px = px - sumx;
			py = py - sumy;
			sumx = 0, sumy = 0;
			for (int m = 0; m < pointNum; m++)
			{
				sumx += S[m] * dstPx[m];
				sumy += S[m] * dstPy[m];
			}
			px = px + sumx;
			py = py + sumy;
			double x_in = CLIP3(px * width, 0, width - 1);
			double y_in = CLIP3(py * height, 0, height - 1);
			//please change interpolation to get better effects.
			int xx = (int)x_in;
			int yy = (int)y_in;
			//pSrcL1 = p + yy * stride;
			//pSrcL2 = pSrcL1 + stride;
			//pos = (xx << 2);//pos = xx*4
			//aa = pSrcL1[pos];
			//bb = pSrcL1[pos + 4];
			//cc = pSrcL2[pos];
			//dd = pSrcL2[pos + 4];
			aa = srcImg.at<cv::Vec3b>(yy, xx);
			bb = srcImg.at<cv::Vec3b>(yy, min(width - 1, xx + 1));
			cc = srcImg.at<cv::Vec3b>(min(height - 1, yy + 1), xx);
			dd = srcImg.at<cv::Vec3b>(min(height - 1, yy + 1), min(width - 1, xx + 1));
			cv::Vec3b r;

			r1 = aa + (bb - aa) * (x_in - xx);
			r2 = cc + (dd - cc) * (x_in - xx);
			r3 = r1 + (r2 - r1) * (y_in - yy);
			result.at<cv::Vec3b>(j, i) = cv::Vec3b(CLIP3(r3[0], 0, 255), CLIP3(r3[1], 0, 255), CLIP3(r3[2], 0, 255));

		}
	}

	free(srcPx);
	free(srcPy);
	free(dstPx);
	free(dstPy);
	free(GammaIJ);
	free(GammaP);
	free(S);
	free(W);
	free(WInv);
	free(tempIJ);
	free(GammaIJT);
	return result;
};

//level:0,不变，>0变大，<0变小
void FaceDeformation::eyes_deformation(cv::Mat &src, dlib::full_object_detection shape, int level)
{
	int pointNum = 0;
	std::vector<int> srcPoint;
	std::vector<int> dstPoint;
	if (level > 0)
		level = 1;
	if (level < 0)
		level = -1;

	if (level != 0)//change eyes size
	{
		pointNum = 13;//固定脸最下一点
		srcPoint.resize(pointNum * 2);
		dstPoint.resize(pointNum * 2);
		for (int i = 0; i < pointNum - 1; i++)
		{
			srcPoint[i * 2] = shape.part(i + 36).x();
			srcPoint[i * 2 + 1] = shape.part(i + 36).y();
			if (i == 1 || i == 2 || i == 7 || i == 8){
				dstPoint[i * 2 + 1] = shape.part(i + 36).y() + level;
			}
			else if (i == 4 || i == 5 || i == 10 || i == 11)
			{
				dstPoint[i * 2 + 1] = shape.part(i + 36).y() - level;
			}
			else
			{
				dstPoint[i * 2 + 1] = shape.part(i + 36).y();
			}
			dstPoint[i * 2] = shape.part(i + 36).x();
			//dstPoint[i * 2 + 1] = shape.part(i + 36).y();
		}
		//srcPoint[(pointNum - 2) * 2] = shape.part(19).x();//固定眉毛的位置，是否需要？
		//srcPoint[(pointNum - 2) * 2+1] = shape.part(19).y();
		//srcPoint[(pointNum - 1) * 2] = shape.part(24).x();
		//srcPoint[(pointNum - 1) * 2 + 1] = shape.part(24).y();
		//dstPoint[(pointNum - 2) * 2] = srcPoint[(pointNum - 2) * 2];
		//dstPoint[(pointNum - 2) * 2+1] = srcPoint[(pointNum - 2) * 2+1];
		//dstPoint[(pointNum - 1) * 2] = srcPoint[(pointNum - 1) * 2];
		//dstPoint[(pointNum - 1) * 2+1] = srcPoint[(pointNum - 1) * 2+1];
		srcPoint[(pointNum - 1) * 2] = shape.part(8).x();//固定脸长度
		srcPoint[(pointNum - 1) * 2 + 1] = shape.part(8).y();
		dstPoint[(pointNum - 1) * 2] = srcPoint[(pointNum - 1) * 2];
		dstPoint[(pointNum - 1) * 2 + 1] = srcPoint[(pointNum - 1) * 2 + 1];
		cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
		//imshow("result",src);
		result.copyTo(src);

	}
}
void FaceDeformation::nose_deformation(cv::Mat src, dlib::full_object_detection shape, int level)
{
	int pointNum = 0;
	std::vector<int> srcPoint;
	std::vector<int> dstPoint;

	if (level != 0)//change nose size
	{
		pointNum = 9;
		srcPoint.resize(pointNum * 2);
		dstPoint.resize(pointNum * 2);

		srcPoint[0] = shape.part(31).x();//鼻子
		srcPoint[1] = shape.part(31).y();
		srcPoint[2] = shape.part(33).x();
		srcPoint[3] = shape.part(33).y();
		srcPoint[4] = shape.part(35).x();
		srcPoint[5] = shape.part(35).y();
		srcPoint[6] = shape.part(3).x();//脸颊位置不变
		srcPoint[7] = shape.part(3).y();
		srcPoint[8] = shape.part(13).x();
		srcPoint[9] = shape.part(13).y();
		srcPoint[10] = shape.part(39).x();//内眼角位置不变
		srcPoint[11] = shape.part(39).y();
		srcPoint[12] = shape.part(42).x();
		srcPoint[13] = shape.part(42).y();
		srcPoint[14] = shape.part(48).x();//嘴角位置不变
		srcPoint[15] = shape.part(48).y();
		srcPoint[16] = shape.part(54).x();
		srcPoint[17] = shape.part(54).y();
		for (int i = 0; i < pointNum * 2; i++)
		{
			dstPoint[i] = srcPoint[i];
		}
		srcPoint[0] = srcPoint[0] - level;
		srcPoint[4] = srcPoint[4] + level;
		cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
		result.copyTo(src);
	}
}

void FaceDeformation::mouth_deformation(cv::Mat src, dlib::full_object_detection shape, int level)
{
	int pointNum = 0;
	std::vector<int> srcPoint;
	std::vector<int> dstPoint;

	if (level != 0)//change mouth size
	{
		pointNum = 8;
		srcPoint.resize(pointNum * 2);
		dstPoint.resize(pointNum * 2);

		srcPoint[0] = shape.part(48).x();//嘴角
		srcPoint[1] = shape.part(48).y();
		srcPoint[2] = shape.part(54).x();
		srcPoint[3] = shape.part(54).y();
		srcPoint[4] = shape.part(35).x();//鼻子位置不变
		srcPoint[5] = shape.part(35).y();
		srcPoint[6] = shape.part(31).x();
		srcPoint[7] = shape.part(31).y();
		srcPoint[8] = shape.part(4).x();//脸颊位置不变
		srcPoint[9] = shape.part(4).y();
		srcPoint[10] = shape.part(6).x();
		srcPoint[11] = shape.part(6).y();
		srcPoint[12] = shape.part(10).x();
		srcPoint[13] = shape.part(10).y();
		srcPoint[14] = shape.part(12).x();
		srcPoint[15] = shape.part(12).y();

		for (int i = 0; i < pointNum * 2; i++)
		{
			dstPoint[i] = srcPoint[i];
		}
		srcPoint[0] = srcPoint[0] - level;
		srcPoint[2] = srcPoint[2] + level;
		cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
		result.copyTo(src);
	}
}

void FaceDeformation::cheek_deformation(cv::Mat src, dlib::full_object_detection shape, int level)
{
	int pointNum = 0;
	std::vector<int> srcPoint;
	std::vector<int> dstPoint;
	if (level != 0)//change cheek size
	{
		pointNum = 5;
		srcPoint.resize(pointNum * 2);
		dstPoint.resize(pointNum * 2);

		srcPoint[0] = shape.part(5).x();//脸颊
		srcPoint[1] = shape.part(5).y();
		srcPoint[2] = shape.part(11).x();
		srcPoint[3] = shape.part(11).y();
		srcPoint[4] = shape.part(0).x();
		srcPoint[5] = shape.part(0).y();
		srcPoint[6] = shape.part(16).x();
		srcPoint[7] = shape.part(16).y();
		srcPoint[8] = shape.part(8).x();
		srcPoint[9] = shape.part(8).y();
		/*srcPoint[10] = shape.part(9).x();
		srcPoint[11] = shape.part(9).y();*/
		//srcPoint[12] = shape.part(10).x();
		//srcPoint[13] = shape.part(10).y();
		//srcPoint[14] = shape.part(12).x();
		//srcPoint[15] = shape.part(12).y();

		for (int i = 0; i < pointNum * 2; i++)
		{
			dstPoint[i] = srcPoint[i];
		}
		srcPoint[0] = srcPoint[0] - level;
		srcPoint[1] = srcPoint[1] + level;
		srcPoint[2] = srcPoint[2] + level;
		srcPoint[3] = srcPoint[3] + level;

		cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
		result.copyTo(src);
	}
}


void FaceDeformation:: deformation(cv::Mat &src)
{
	int pointNum = 0;
	string filename = "G:\\4.jpg";
	//cv::Mat src = cv::imread(filename);
	int eye_change = 0;//1:bigger,2:smaller，0:no change
	int nose_change = 0;
	int mouth_change = 0;
	int cheek_change = 0;
	//int dstPoint[] = { 226, 533, 510, 544 };
	//int srcPoint[] = { 226, 531, 510, 543 };
	std::vector<int> srcPoint;
	std::vector<int> dstPoint;

	try
	{
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

		array2d<rgb_pixel> img;
		load_image(img, filename);

		std::vector<dlib::rectangle> dets = detector(img);
		if (dets.size() < 1)//not detect face
			return;
		full_object_detection shape = sp(img, dets[0]);//only detect a face

		//draw point in the face
		/*for (int i = 0; i < pointNum; i++) {
		circle(src, cvPoint(shape.part(i).x(), shape.part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
		putText(src, to_string(i), cvPoint(shape.part(i).x(), shape.part(i).y()), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 4);
		}*/
		/*imwrite("G:\\11.jpg",src);
		return 0;*/

		//鼻子大小调整
		if (shape.part(42).x() - shape.part(39).x()<shape.part(35).x() - shape.part(31).x())//内眼角和鼻翼大小比较
		{
			nose_change = -3;
		}
		else{
			nose_change = -1;
		}
		cheek_change = 0;//-3
		eye_change = 1;
		nose_change = 0;
		if (eye_change != 0)//change eyes size
		{
			pointNum = 13;//固定脸最下一点
			srcPoint.resize(pointNum * 2);
			dstPoint.resize(pointNum * 2);
			for (int i = 0; i < pointNum - 1; i++)
			{
				srcPoint[i * 2] = shape.part(i + 36).x();
				srcPoint[i * 2 + 1] = shape.part(i + 36).y();
				if (i == 1 || i == 2 || i == 7 || i == 8){
					dstPoint[i * 2 + 1] = shape.part(i + 36).y() + eye_change;
				}
				else if (i == 4 || i == 5 || i == 10 || i == 11)
				{
					dstPoint[i * 2 + 1] = shape.part(i + 36).y() - eye_change;
				}
				else
				{
					dstPoint[i * 2 + 1] = shape.part(i + 36).y();
				}
				dstPoint[i * 2] = shape.part(i + 36).x();
				//dstPoint[i * 2 + 1] = shape.part(i + 36).y();
			}
			//srcPoint[(pointNum - 2) * 2] = shape.part(19).x();//固定眉毛的位置，是否需要？
			//srcPoint[(pointNum - 2) * 2+1] = shape.part(19).y();
			//srcPoint[(pointNum - 1) * 2] = shape.part(24).x();
			//srcPoint[(pointNum - 1) * 2 + 1] = shape.part(24).y();
			//dstPoint[(pointNum - 2) * 2] = srcPoint[(pointNum - 2) * 2];
			//dstPoint[(pointNum - 2) * 2+1] = srcPoint[(pointNum - 2) * 2+1];
			//dstPoint[(pointNum - 1) * 2] = srcPoint[(pointNum - 1) * 2];
			//dstPoint[(pointNum - 1) * 2+1] = srcPoint[(pointNum - 1) * 2+1];
			srcPoint[(pointNum - 1) * 2] = shape.part(8).x();//固定脸长度
			srcPoint[(pointNum - 1) * 2 + 1] = shape.part(8).y();
			dstPoint[(pointNum - 1) * 2] = srcPoint[(pointNum - 1) * 2];
			dstPoint[(pointNum - 1) * 2 + 1] = srcPoint[(pointNum - 1) * 2 + 1];
			cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
			imshow("result",result);
			result.copyTo(src);

		}

		if (nose_change != 0)//change nose size
		{
			pointNum = 9;
			srcPoint.resize(pointNum * 2);
			dstPoint.resize(pointNum * 2);

			srcPoint[0] = shape.part(31).x();//鼻子
			srcPoint[1] = shape.part(31).y();
			srcPoint[2] = shape.part(33).x();
			srcPoint[3] = shape.part(33).y();
			srcPoint[4] = shape.part(35).x();
			srcPoint[5] = shape.part(35).y();
			srcPoint[6] = shape.part(3).x();//脸颊位置不变
			srcPoint[7] = shape.part(3).y();
			srcPoint[8] = shape.part(13).x();
			srcPoint[9] = shape.part(13).y();
			srcPoint[10] = shape.part(39).x();//内眼角位置不变
			srcPoint[11] = shape.part(39).y();
			srcPoint[12] = shape.part(42).x();
			srcPoint[13] = shape.part(42).y();
			srcPoint[14] = shape.part(48).x();//嘴角位置不变
			srcPoint[15] = shape.part(48).y();
			srcPoint[16] = shape.part(54).x();
			srcPoint[17] = shape.part(54).y();
			for (int i = 0; i < pointNum * 2; i++)
			{
				dstPoint[i] = srcPoint[i];
			}
			srcPoint[0] = srcPoint[0] - nose_change;
			srcPoint[4] = srcPoint[4] + nose_change;
			cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
			result.copyTo(src);
		}

		if (mouth_change != 0)//change mouth size
		{
			pointNum = 8;
			srcPoint.resize(pointNum * 2);
			dstPoint.resize(pointNum * 2);

			srcPoint[0] = shape.part(48).x();//嘴角
			srcPoint[1] = shape.part(48).y();
			srcPoint[2] = shape.part(54).x();
			srcPoint[3] = shape.part(54).y();
			srcPoint[4] = shape.part(35).x();//鼻子位置不变
			srcPoint[5] = shape.part(35).y();
			srcPoint[6] = shape.part(31).x();
			srcPoint[7] = shape.part(31).y();
			srcPoint[8] = shape.part(4).x();//脸颊位置不变
			srcPoint[9] = shape.part(4).y();
			srcPoint[10] = shape.part(6).x();
			srcPoint[11] = shape.part(6).y();
			srcPoint[12] = shape.part(10).x();
			srcPoint[13] = shape.part(10).y();
			srcPoint[14] = shape.part(12).x();
			srcPoint[15] = shape.part(12).y();

			for (int i = 0; i < pointNum * 2; i++)
			{
				dstPoint[i] = srcPoint[i];
			}
			srcPoint[0] = srcPoint[0] - mouth_change;
			srcPoint[2] = srcPoint[2] + mouth_change;
			cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
			result.copyTo(src);
		}

		if (cheek_change != 0)//change cheek size
		{
			pointNum = 5;
			srcPoint.resize(pointNum * 2);
			dstPoint.resize(pointNum * 2);

			srcPoint[0] = shape.part(5).x();//脸颊
			srcPoint[1] = shape.part(5).y();
			srcPoint[2] = shape.part(11).x();
			srcPoint[3] = shape.part(11).y();
			srcPoint[4] = shape.part(0).x();
			srcPoint[5] = shape.part(0).y();
			srcPoint[6] = shape.part(16).x();
			srcPoint[7] = shape.part(16).y();
			srcPoint[8] = shape.part(8).x();
			srcPoint[9] = shape.part(8).y();
			/*srcPoint[10] = shape.part(9).x();
			srcPoint[11] = shape.part(9).y();*/
			//srcPoint[12] = shape.part(10).x();
			//srcPoint[13] = shape.part(10).y();
			//srcPoint[14] = shape.part(12).x();
			//srcPoint[15] = shape.part(12).y();

			for (int i = 0; i < pointNum * 2; i++)
			{
				dstPoint[i] = srcPoint[i];
			}
			srcPoint[0] = srcPoint[0] - cheek_change;
			srcPoint[1] = srcPoint[1] + cheek_change;
			srcPoint[2] = srcPoint[2] + cheek_change;
			srcPoint[3] = srcPoint[3] + cheek_change;

			cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
			result.copyTo(src);
		}

	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}


	//cv::Mat result = f_MLSR(src, srcPoint, dstPoint, pointNum);
	imwrite("G:\\44.jpg", src);

	//return 0;
}


