#include "StyleTransfer.h"


StyleTransfer::StyleTransfer()
{
}


StyleTransfer::~StyleTransfer()
{
}

double NearColor(cv::Vec3d color1, cv::Vec3d color2){
	//BGR to YUV
	double Y1 = 0.299*color1[2] + 0.587*color1[1] + 0.114*color1[0];
	double U1 = -0.147*color1[2] - 0.289*color1[1] + 0.436*color1[0];
	double V1 = 0.615*color1[2] - 0.515*color1[1] - 0.100*color1[0];

	double Y2 = 0.299*color2[2] + 0.587*color2[1] + 0.114*color2[0];
	double U2 = -0.147*color2[2] - 0.289*color2[1] + 0.436*color2[0];
	double V2 = 0.615*color2[2] - 0.515*color2[1] - 0.100*color2[0];

	double distance = sqrt((U1 - U2)*(U1 - U2) + (V1 - V2)*(V1 - V2) + (Y1 - Y2)*(Y1 - Y2));
	return distance;
}

cv::Mat StyleTransfer::nearColorEdgeDetector(cv::Mat src){
	//输入32FC3的原图 输出32FC4的边缘图
	double nearcolor_value = 0.1;//相近颜色阈值，越大检测的边缘需要的颜色差越大,原为0.1
	cv::Mat output = cv::Mat::zeros(src.size(), CV_32FC4);
	int height = src.size().height;
	int width = src.size().width;
	for (int row = 1; row < height - 1; row++) {
		for (int col = 1; col < width - 1; col++) {
			cv::Vec3f colorC = src.at<cv::Vec3f>(row, col);
			cv::Vec3f colorL = src.at<cv::Vec3f>(row, col - 1);
			cv::Vec3f colorR = src.at<cv::Vec3f>(row, col + 1);
			cv::Vec3f colorT = src.at<cv::Vec3f>(row - 1, col);
			cv::Vec3f colorB = src.at<cv::Vec3f>(row + 1, col);
			output.at<cv::Vec4f>(row, col) = cv::Vec4f(0, 0, 0, 0);
			if (NearColor(colorL, colorR) >= nearcolor_value) {
				if (NearColor(colorL, colorC)<NearColor(colorR, colorC)) {
					output.at<cv::Vec4f>(row, col) = cv::Vec4f(1.0, 0.0, 0.0, 1.0);//other color:right
				}
				else{
					output.at<cv::Vec4f>(row, col) = cv::Vec4f(-1.0, 0.0, 0.0, 1.0);//left
				}
			}
			if (NearColor(colorT, colorB) >= nearcolor_value) {
				if (NearColor(colorT, colorC)<NearColor(colorB, colorC)) {
					output.at<cv::Vec4f>(row, col) = cv::Vec4f(0.0, 1.0, 0.0, 1.0);//bottom
				}
				else{
					output.at<cv::Vec4f>(row, col) = cv::Vec4f(0.0, -1.0, 0.0, 1.0);//top
				}
			}
		}
	}

	return output;
}

cv::Mat StyleTransfer::largeGradient(cv::Mat src){
	//输入 32FC3的src原图 输出32FC3的梯度
	cv::Mat output = cv::Mat::zeros(src.size(), CV_32FC3);
	cv::Mat src2;
	src.copyTo(src2);
	cv::GaussianBlur(src2, src2, cv::Size(15, 15), 6.0, 0, cv::BORDER_DEFAULT);//注意size大小
	int height = src2.size().height;
	int width = src2.size().width;
	int kernel = 3;
	for (int row = kernel; row < height - kernel; row++) {
		for (int col = kernel; col < width - kernel; col++) {
			cv::Vec3f u = cv::Vec3f(0.0, 0.0, 0.0), v = cv::Vec3f(0.0, 0.0, 0.0);
			for (int i = 1; i <= kernel; i++) {
				for (int j = -kernel; j <= kernel; j++) {
					/*	u += src.at<cv::Vec3f>(row + i, col + j) - src.at<cv::Vec3f>(row - i, col + j);
					v += src.at<cv::Vec3f>(row + j, col + i) - src.at<cv::Vec3f>(row + j, col - i);*/
					u += src2.at<cv::Vec3f>(row + j, col + i) - src2.at<cv::Vec3f>(row + j, col - i);
					v += src2.at<cv::Vec3f>(row + i, col + j) - src2.at<cv::Vec3f>(row - i, col + j);
				}
			}
			float norm = (kernel * (kernel * 2 + 1))*0.4;
			v /= norm;
			u /= norm;
			output.at<cv::Vec3f>(row, col)[0] = (u[0] + u[1] + u[2]) / 3.0;
			output.at<cv::Vec3f>(row, col)[1] = (v[0] + v[1] + v[2]) / 3.0;
			output.at<cv::Vec3f>(row, col)[2] = 0.0;
		}
	}

	return output;
}
float HueDistance(float hue1, float hue2){
	if (hue1 >= hue2)
		return MIN(hue1 - hue2, 360.0 - hue1 + hue2);
	else
		return MIN(hue2 - hue1, 360.0 - hue2 + hue1);
}
float mod(float x, float y)
{
	return x - y * floor(x / y);
}
cv::Vec3f getHLS(cv::Vec3f color){
	int H, L, S;
	float M = MAX(MAX(color[0], color[1]), color[2]);
	float m = MIN(MIN(color[0], color[1]), color[2]);
	float C = M - m;
	if (C == 0.0) {
		H = -1.0;
	}
	else if (M == color[2]) {
		H = mod((color[1] - color[0]) / C, 6.0);
	}
	else if (M == color[1]){
		H = (color[0] - color[2]) / C + 2.0;
	}
	else{
		H = (color[2] - color[1]) / C + 4.0;
	}
	H *= 60.0;
	L = 0.5 * (M + m);
	if (C == 0.0) {
		S = 0.0;
	}
	else{
		S = C / (1.0 - fabs(2.0 * L - 1.0));
	}
	return cv::Vec3f(H, L, S);
}
cv::Mat StyleTransfer::edgeTypeDetect(cv::Mat src, cv::Mat edge, cv::Mat gradients, cv::Mat mask){
	//输入 32FC3原图 32FC4边缘  32FC3梯度 输出32FC4边缘类型
	cv::Mat output;
	cv::Mat HLSedge;
	int height = src.size().height;
	int width = src.size().width;
	edge.copyTo(HLSedge);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (HLSedge.at<cv::Vec4f>(row, col)[3] == 1.0) {
				int drow = (int)edge.at<cv::Vec4f>(row, col)[1];//修改
				int dcol = (int)edge.at<cv::Vec4f>(row, col)[0];

				HLSedge.at<cv::Vec4f>(row, col)[0] = src.at<cv::Vec3f>(row + drow, col + dcol)[0];
				HLSedge.at<cv::Vec4f>(row, col)[1] = src.at<cv::Vec3f>(row + drow, col + dcol)[1];
				HLSedge.at<cv::Vec4f>(row, col)[2] = src.at<cv::Vec3f>(row + drow, col + dcol)[2];
			}

		}
	}
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (HLSedge.at<cv::Vec4f>(row, col)[3] == 1.0) {
				cv::Vec3f hlsimage = getHLS(src.at<cv::Vec3f>(row, col));
				cv::Vec3f hlsedge = getHLS(HLSedge.at<cv::Vec3f>(row, col));
				float gradient_x = gradients.at<cv::Vec3f>(row, col)[0];
				float gradient_y = gradients.at<cv::Vec3f>(row, col)[1];
				bool nearhue = HueDistance(hlsedge[0], hlsimage[0]) < 90.0;

				if (mask.at<uchar>(row, col)>245)	//>245	
				{
					HLSedge.at<cv::Vec4f>(row, col)[3] = 1.0;//皮肤不添加湿画区域
				}
				else if (fabs(gradient_x) < 0.1 && fabs(gradient_y) < 0.1) {
					HLSedge.at<cv::Vec4f>(row, col)[3] = 1.0;//不添加湿画区域
				}
				else if (HueDistance(hlsedge[0], hlsimage[0]) < 20.0){
					HLSedge.at<cv::Vec4f>(row, col)[3] = 0.5;//湿画区域
				}
				/*else if (HueDistance(hlsedge[0], hlsimage[0]) < 90.0&&fabs(hlsimage[1] - hlsedge[1])<0.2)
				{
					HLSedge.at<cv::Vec4f>(row, col)[3] = 0.5;
				}
				else if ((hlsimage[2] > 0.3 && hlsedge[2] > 0.3 && !nearhue) || fabs(hlsimage[1] - hlsedge[1]) > 0.6 || (fabs(gradient_x) < 0.1 && fabs(gradient_y) < 0.1)){
					HLSedge.at<cv::Vec4f>(row, col)[3] = 1.0;
				}*/
				else{
					HLSedge.at<cv::Vec4f>(row, col)[3] = 0.0;
				}
				//HLSedge.at<cv::Vec4f>(row, col)[3] = 0.5;//湿画区域
			}
		}
	}

	HLSedge.copyTo(output);

	return output;
}

cv::Mat StyleTransfer::wetPre(cv::Mat input, cv::Mat edgeType, cv::Mat gradients, cv::Mat randomNoiseImage)
{

	//输入32FC3 edgetype 32fc4
	int half_width = 6;//原为6
	float noise_control = 0.4;//原为0.4，可控制小颗粒生成的数目和远近
	
	cv::Mat output = cv::Mat(input.size(), CV_32FC4);
	int height = input.size().height;
	int width = input.size().width;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			cv::Vec3f color = input.at<cv::Vec3f>(row, col);
			float in_out = 0.0, dist = 0.0;
			float nearestdist = 100.0;//原为100.0
			cv::Vec3f other_color;
			for (int i = -half_width; i <= half_width; i++) {
				for (int j = -half_width; j < half_width; j++) {
					if (row + i < 0 || row + i >= height || col + j < 0 || col + j >= width) {
						continue;
					}
					if (edgeType.at<cv::Vec4f>(row + i, col + j)[3] == 0.5) {
						dist = sqrtf(i*i + j*j);
						if (dist < nearestdist) {
							nearestdist = dist;
							in_out = 1.0;
							other_color = cv::Vec3f(edgeType.at<cv::Vec4f>(row + i, col + j)[0], edgeType.at<cv::Vec4f>(row + i, col + j)[1], edgeType.at<cv::Vec4f>(row + i, col + j)[2]);
						}
					}
				}
			}
			nearestdist /= sqrtf(2.0) * half_width;
			if (in_out == 1.0){
				//if (other_color[0] + other_color[1] + other_color[2] > color[0] + color[1] + color[2]) {//浅
				//	in_out = 0.5;
				//}
				if (0.299*color[2] + 0.587*color[1] + 0.114*color[0]<0.299*other_color[2] + 0.587*other_color[1] + 0.114*other_color[0])
					in_out = 0.5;
			}
			output.at<cv::Vec4f>(row, col) = cv::Vec4f(color[0], color[1], color[2], in_out);

			if (in_out == 1.0){//湿画区域
				int rown = (int)((float)(row)* randomNoiseImage.size().height / ((float)(height)));
				int coln = (int)((float)(col)* randomNoiseImage.size().width / ((float)(width)));
				cv::Vec3b colornoise = randomNoiseImage.at<cv::Vec3b>(rown, coln);
				//float noise = (float)(colornoise[0] + colornoise[1] + colornoise[2]) / (3.0 * 255.0);
				//float noise = (float)(colornoise[1]) / (255.0);
				float noise = 200.0 / (255.0);
				if (noise * (1 - nearestdist) * 2 > noise_control * 2) {
					output.at<cv::Vec4f>(row, col) = cv::Vec4f(other_color[0], other_color[1], other_color[2], in_out);//修改
				}
			}
			//控制里面一圈是否被滤波
			/*if (output.at<cv::Vec4f>(row, col)[3] == 0.5)
			output.at<cv::Vec4f>(row, col)[3] = 1.0;*/
		}
	}
	return output;
}
float WeightCal(float x, float y, cv::Vec3f theta, float maxA, float maxB){
	float y2 = -theta[1] * x + theta[0] * y;
	float x2 = theta[0] * x + theta[1] * y;
	return MAX(1.0 - (x2 * x2 / maxA + y2 * y2 / maxB), 0.0);

}
cv::Mat StyleTransfer::wetFilter(cv::Mat wetpre, cv::Mat gradients){
	cv::Mat output = cv::Mat(wetpre.size(), CV_32FC4);
	wetpre.copyTo(output);
	int width = wetpre.size().width;
	int height = wetpre.size().height;

	int wet_degree = 15;//原为10
	int maxR = int(ceil(float(wet_degree) / 3.0));//大于等于的最小整数,3
	float maxA = float((maxR + 1) * (maxR + 1));//半长轴，16
	//float eccentricity = MAX(1.0, 16.0 * MIN(1.0, 20.0 / float(wet_degree)));//16
	float k = wet_degree / 80.0;
	k = MIN(MAX(0.25, k), 1.0);
	float maxB = maxA * (k*k);//半短轴，1
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			float weightSum = 1.0;
			int weight_inout = 0;
			cv::Vec3f direction = gradients.at<cv::Vec3f>(row, col);
			
			if (fabs(direction[0]) < 0.000001&&fabs(direction[1]) < 0.000001)
			{

			}
			else if (fabs(direction[0]) > fabs(direction[1]))
			{

				direction[1] = direction[1] / fabs(direction[0]);
				if (direction[0] < 0)
					direction[0] = -1.0;
				else
					direction[0] = 1.0;
			}
			else
			{

				direction[0] = direction[0] / fabs(direction[1]);
				if (direction[1] < 0)
					direction[1] = -1.0;
				else
					direction[1] = 1.0;
			}
			cv::Vec4f cur_color = wetpre.at<cv::Vec4f>(row, col);
			cv::Vec3f colorSum = cv::Vec3f(cur_color[0], cur_color[1], cur_color[2]);
			if (cur_color[3] < 1.0) {//not wet zone
				//output.at<cv::Vec4f>(row, col) = cur_color;
				continue;
			}
			for (int i = 1; i <= maxR; i++) {
				float weight = WeightCal(0.0, float(i), direction, maxA, maxB);
				int row2 = row + i;
				if (row2 < height && row2 >= 0) {
					cv::Vec4f color = wetpre.at<cv::Vec4f>(row2, col);
					colorSum += weight * cv::Vec3f(color[0], color[1], color[2]);
					weightSum += weight;
					if (color[3] == 1.0) {
						weight_inout++;
					}
				}
				row2 = row - i;
				if (row2 < height && row2 >= 0) {
					cv::Vec4f color = wetpre.at<cv::Vec4f>(row2, col);
					colorSum += weight * cv::Vec3f(color[0], color[1], color[2]);
					weightSum += weight;
					if (color[3] == 1.0) {
						weight_inout++;
					}
				}
			}
			for (int i = 1; i <= maxR; i++) {
				for (int j = -maxR + 1; j < maxR; j++) {
					float weight = WeightCal(float(i), float(j), direction, maxA, maxB);
					int row2 = row + j;
					int col2 = col + i;
					if (row2 < height && row2 >= 0 && col2 < width && col2 >= 0) {
						cv::Vec4f color = wetpre.at<cv::Vec4f>(row2, col2);
						colorSum += weight * cv::Vec3f(color[0], color[1], color[2]);
						weightSum += weight;
						if (color[3] == 1.0) {
							weight_inout++;
						}
					}
					row2 = row - j;
					col2 = col - i;
					if (row2 < height && row2 >= 0 && col2 < width && col2 >= 0) {
						cv::Vec4f color = wetpre.at<cv::Vec4f>(row2, col2);
						colorSum += weight * cv::Vec3f(color[0], color[1], color[2]);
						weightSum += weight;
						if (color[3] == 1.0) {
							weight_inout++;
						}
					}
				}
			}
			colorSum /= weightSum;
			if (weight_inout > 0) {
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorSum[0], colorSum[1], colorSum[2], cur_color[3]);
			}
			else{
				output.at<cv::Vec4f>(row, col) = cur_color;
			}
			
		}
	}
	
	return output;
}
float clamp(float a, float m, float M){
	return MIN(MAX(a, m), M);
}

cv::Vec4f clamp(cv::Vec4f a, cv::Vec4f m, cv::Vec4f M){
	return cv::Vec4f(clamp(a[0], m[0], M[0]), clamp(a[1], m[1], M[1]), clamp(a[2], m[2], M[2]), clamp(a[3], m[3], M[3]));
}
double dist(cv::Point a, cv::Point b)
{
	return sqrt(pow((double)(a.x - b.x), 2) + pow((double)(a.y - b.y), 2));
}
float dist(cv::Vec3f a, cv::Vec3f b){
	return sqrtf(powf(a[0] - b[0], 2) + powf(a[1] - b[1], 2) + powf(a[1] - b[1], 2));
}

float dist(cv::Vec4f a, cv::Vec4f b){
	return sqrtf(powf(a[0] - b[0], 2) + powf(a[1] - b[1], 2) + powf(a[1] - b[1], 2));
}

cv::Mat StyleTransfer::edgeDarken(cv::Mat distortImg, cv::Mat wetfilterImg, cv::Mat mask, double level, int flag){
	cv::Mat output;
	distortImg.copyTo(output);
	double edgedarkening_scale = level;//原为2.5

	float cutt = 0.1;
	int height = distortImg.size().height;
	int width = distortImg.size().width;
	for (int row = 3; row < height - 3; row++) {
		for (int col = 3; col < width - 3; col++) {
			if (flag == 1 && mask.at<uchar>(row, col) < 200)
			{
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(0.0,0.0,0.0,0.0);
				continue;
			}
				
			float darkening_scale = edgedarkening_scale;
			if (flag == 0 && mask.at<uchar>(row, col)>250)//皮肤部分
			{
				darkening_scale = 1.5;
			}
			cv::Vec4f color = distortImg.at<cv::Vec4f>(row, col);
			cv::Vec4f top_color = distortImg.at<cv::Vec4f>(row - 1, col);
			cv::Vec4f bottom_color = distortImg.at<cv::Vec4f>(row + 1, col);
			cv::Vec4f left_color = distortImg.at<cv::Vec4f>(row, col - 1);
			cv::Vec4f right_color = distortImg.at<cv::Vec4f>(row, col + 1);
			cv::Vec4f grad_row_color = bottom_color - top_color;
			cv::Vec4f grad_col_color = right_color - left_color;
			float grad_value_hard = (fabs(grad_col_color[0]) + fabs(grad_col_color[1]) + fabs(grad_col_color[2]) + fabs(grad_row_color[0]) + fabs(grad_row_color[1]) + fabs(grad_row_color[2])) / 6.0;
			grad_value_hard = clamp(grad_value_hard, 0.0, cutt);
			float grad_value = 0.0;
			//calculate grad color
			if (wetfilterImg.at<cv::Vec4f>(row, col)[3] == 0.0) {
				cv::Vec4f grad_row = cv::Vec4f(0.0, 0.0, 0.0, 0.0);
				cv::Vec4f grad_col = cv::Vec4f(0.0, 0.0, 0.0, 0.0);
				cv::Vec4f cutv1 = cv::Vec4f(-cutt, -cutt, -cutt, -cutt);
				cv::Vec4f cutv2 = cv::Vec4f(cutt, cutt, cutt, cutt);
				for (int i = 1; i < 4; i++) {
					for (int j = 1; j < 4; j++) {
						if ((wetfilterImg.at<cv::Vec4f>(row + i, col + j)[3] == 0.0) && (wetfilterImg.at<cv::Vec4f>(row - i, col + j)[3] == 0.0)) {
							grad_row += (4 - i) * clamp(distortImg.at<cv::Vec4f>(row + i, col + j) - distortImg.at<cv::Vec4f>(row - i, col + j), cutv1, cutv2);
						}
						if ((wetfilterImg.at<cv::Vec4f>(row + i, col - j)[3] == 0.0) && (wetfilterImg.at<cv::Vec4f>(row - i, col - j)[3] == 0.0)) {
							grad_row += (4 - i) * clamp(distortImg.at<cv::Vec4f>(row + i, col - j) - distortImg.at<cv::Vec4f>(row - i, col - j), cutv1, cutv2);
						}

						if ((wetfilterImg.at<cv::Vec4f>(row + i, col + j)[3] == 0.0) && (wetfilterImg.at<cv::Vec4f>(row + i, col - j)[3] == 0.0)) {
							grad_col += (4 - j) * clamp(distortImg.at<cv::Vec4f>(row + i, col + j) - distortImg.at<cv::Vec4f>(row + i, col - j), cutv1, cutv2);
						}
						if ((wetfilterImg.at<cv::Vec4f>(row - i, col + j)[3] == 0.0) && (wetfilterImg.at<cv::Vec4f>(row - i, col - j)[3] == 0.0)) {
							grad_col += (4 - j) * clamp(distortImg.at<cv::Vec4f>(row - i, col + j) - distortImg.at<cv::Vec4f>(row - i, col - j), cutv1, cutv2);
						}
					}
					if ((wetfilterImg.at<cv::Vec4f>(row + i, col)[3] == 0.0) && (wetfilterImg.at<cv::Vec4f>(row - i, col)[3] == 0.0)) {
						grad_row += (4 - i) * clamp(distortImg.at<cv::Vec4f>(row + i, col) - distortImg.at<cv::Vec4f>(row - i, col), cutv1, cutv2);
					}
					if ((wetfilterImg.at<cv::Vec4f>(row, col + i)[3] == 0.0) && (wetfilterImg.at<cv::Vec4f>(row, col - i)[3] == 0.0)) {
						grad_col += (4 - i) * clamp(distortImg.at<cv::Vec4f>(row, col + i) - distortImg.at<cv::Vec4f>(row, col - i), cutv1, cutv2);
					}
				}
				grad_row /= 42.0;
				grad_col /= 42.0;
				grad_value = 0.5 * (sqrtf(pow(grad_row[0], 2) + pow(grad_row[1], 2) + pow(grad_row[2], 2)) + sqrtf(pow(grad_col[0], 2) + pow(grad_col[1], 2) + pow(grad_col[2], 2)));
				if (grad_row[0] + grad_row[1] + grad_row[2] > 0.0) {
					if (dist(distortImg.at<cv::Vec4f>(row, col), distortImg.at<cv::Vec4f>(row + 3, col)) < dist(distortImg.at<cv::Vec4f>(row, col), distortImg.at<cv::Vec4f>(row - 3, col))) {
						grad_value = 0.0;
					}
				}
				if (grad_col[0] + grad_col[1] + grad_col[2] > 0.0) {
					if (dist(distortImg.at<cv::Vec4f>(row, col), distortImg.at<cv::Vec4f>(row, col + 3)) < dist(distortImg.at<cv::Vec4f>(row, col), distortImg.at<cv::Vec4f>(row, col - 3))) {
						grad_value = 0.0;
					}
				}
			}
			darkening_scale = clamp(darkening_scale * (grad_value + grad_value_hard), 0.0, 1.0);
			cv::Vec3f colorc = cv::Vec3f(color[0], color[1], color[2]);
			//cv::Vec3f c = colorc - darkening_scale * (colorc - colorc.cross(colorc));
			cv::Vec3f c = colorc - darkening_scale * (colorc - cv::Vec3f(colorc[0] * colorc[0], colorc[1] * colorc[1], colorc[2] * colorc[2]));
			output.at<cv::Vec4f>(row, col) = cv::Vec4f(c[0], c[1], c[2], color[3]);
		}
	}
	
	return output;
}

cv::Mat StyleTransfer::pigmentVariation(cv::Mat input, cv::Mat randomNoiseImg, cv::Mat perlinNoiseImage, cv::Mat paper, cv::Mat mask, double level, int flag){
	cv::Mat output;
	input.copyTo(output);
	cv::Mat randomNoise, perlinNoise;
	
	randomNoiseImg.convertTo(randomNoise, CV_32FC3, 1 / 255.0);
	perlinNoiseImage.convertTo(perlinNoise, CV_32FC3, 1 / 255.0);
	resize(perlinNoise, perlinNoise, cv::Size(input.cols, input.rows), CV_INTER_CUBIC);
	resize(randomNoise, randomNoise, cv::Size(input.cols, input.rows), CV_INTER_CUBIC);
	//paperTexture.convertTo(paperTexture, CV_32FC3, 1 / 255.0);
	float dispersionBeta ;//1.5,3.0
	float flowBeta ;//2.0,4.0
	float paperBeta ;
	int height = input.size().height;
	int width = input.size().width;
	//imwrite("random.jpg",randomNoise);
	//level = 1.0;//frd
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//if (mask.at<uchar>(row, col)>252)//皮肤部分
			//{
			//	dispersionBeta = 1.5;
			//	flowBeta = 2.5;
			//}
			if (flag == 1 && mask.at<uchar>(row, col) < 200)
			{
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(0.0, 0.0, 0.0, 0.0);
				continue;
			}
			
			if (flag == 0){
				uchar mask_value = mask.at<uchar>(row, col);
				dispersionBeta = level - mask_value / 255.0*0.5;
				flowBeta = level - mask_value / 255.0*0.5;
				/*dispersionBeta = level;
				flowBeta = level;*/
			}
			else
			{
				dispersionBeta = level ;
				flowBeta = level ;
			}
			cv::Vec4f color = input.at<cv::Vec4f>(row, col);
			cv::Vec3f colorc = cv::Vec3f(color[0], color[1], color[2]);
			//float random = (1.0 - randomNoise.at<cv::Vec3f>(row*random_h / height, col*random_w / width)[2]) / 3.0;//修改
			//float random = (1.0 - randomNoise.at<cv::Vec3f>(row, col)[2]) / 3.0;//修改
			float random = randomNoise.at<cv::Vec3f>(row, col)[0];//修改
			colorc = colorc - dispersionBeta * (random / 2 - 0.1) * (colorc - cv::Vec3f(colorc[0] * colorc[0], colorc[1] * colorc[1], colorc[2] * colorc[2]));//-0.1
			//float perlin = perlinNoise.at<cv::Vec3f>(row*perlin_h / height, col*perlin_w / width)[0];
			float perlin = perlinNoise.at<cv::Vec3f>(row, col)[0];
			colorc = colorc - flowBeta* (perlin - 0.5) * (colorc - cv::Vec3f(colorc[0] * colorc[0], colorc[1] * colorc[1], colorc[2] * colorc[2]));//-0.5
			
			//float papernoise = paperTexture.at<cv::Vec3f>(row*paper_h / height, col*paper_w / width)[0];
			//colorc = colorc - paperBeta * (papernoise - 0.8) * (colorc - Vec3f(colorc[0] * colorc[0], colorc[1] * colorc[1], colorc[2] * colorc[2]));//-0.5改成了0.8,0.8会使整体亮度变大
			output.at<cv::Vec4f>(row, col) = cv::Vec4f(fabs(colorc[0]), fabs(colorc[1]), fabs(colorc[2]), 1.0);
			//output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
		}
	}
	return output;
}
cv::Vec4f mod289(cv::Vec4f x){
	x[0] = x[0] - floor(x[0] * (1.0 / 289.0)) * 289.0;
	x[1] = x[1] - floor(x[1] * (1.0 / 289.0)) * 289.0;
	x[2] = x[2] - floor(x[2] * (1.0 / 289.0)) * 289.0;
	x[3] = x[3] - floor(x[3] * (1.0 / 289.0)) * 289.0;
	return x;
}
cv::Vec4f permute(cv::Vec4f x){
	return mod289(((x * 34.0) + cv::Vec4f(1.0, 1.0, 1.0, 1.0)) * x);
}
cv::Vec4f taylorInvSqrt(cv::Vec4f r){
	float a = 1.79284291400159;
	float b = 0.85373472095314;
	return (cv::Vec4f(a, a, a, a) - b * r);
}
cv::Vec4f floor(cv::Vec4f x){
	return cv::Vec4f(floor(x[0]), floor(x[1]), floor(x[2]), floor(x[3]));
}
cv::Vec4f fract(cv::Vec4f x){
	return cv::Vec4f(x[0] - floor(x[0]), x[1] - floor(x[1]), x[2] - floor(x[2]), x[3] - floor(x[3]));
}
cv::Vec2f mix(cv::Vec2f a, cv::Vec2f b, float c){
	return a*(1 - c) + b*c;
}
float mix(float a, float b, float c){
	return a*(1 - c) + b*c;
}
float fade(float t){
	return t*t*t*(t*(t*6.0 - 15.0) + 10.0);
}

float cnoise(cv::Vec2f P){
	cv::Vec4f Pi = cv::Vec4f(floor(P[0]), floor(P[1]), floor(P[0]), floor(P[1])) + cv::Vec4f(0.0, 0.0, 1.0, 1.0);
	cv::Vec4f Pf = cv::Vec4f(P[0] - floor(P[0]), P[1] - floor(P[1]), P[0] - floor(P[0]), P[1] - floor(P[1])) - cv::Vec4f(0.0, 0.0, 1.0, 1.0);
	Pi = mod289(Pi);
	cv::Vec4f ix = cv::Vec4f(Pi[0], Pi[2], Pi[0], Pi[2]);
	cv::Vec4f iy = cv::Vec4f(Pi[1], Pi[3], Pi[1], Pi[3]);
	cv::Vec4f fx = cv::Vec4f(Pf[0], Pf[2], Pf[0], Pf[2]);
	cv::Vec4f fy = cv::Vec4f(Pf[1], Pf[3], Pf[1], Pf[3]);
	cv::Vec4f i = permute(permute(ix) + iy);
	cv::Vec4f gx = fract(i*(1.0 / 41.0)) * 2.0 - cv::Vec4f(1.0, 1.0, 1.0, 1.0);
	cv::Vec4f gy = cv::Vec4f(fabs(gx[0]) - 0.5, fabs(gx[1]) - 0.5, fabs(gx[2]) - 0.5, fabs(gx[3]) - 0.5);
	cv::Vec4f tx = floor(gx + cv::Vec4f(0.5, 0.5, 0.5, 0.5));
	gx = gx - tx;
	cv::Vec2f g00 = cv::Vec2f(gx[0], gy[0]);
	cv::Vec2f g10 = cv::Vec2f(gx[1], gy[1]);
	cv::Vec2f g01 = cv::Vec2f(gx[2], gy[2]);
	cv::Vec2f g11 = cv::Vec2f(gx[3], gy[3]);
	cv::Vec4f norm = taylorInvSqrt(cv::Vec4f(g00.dot(g00), g01.dot(g01), g10.dot(g10), g11.dot(g11)));
	g00 *= norm[0];
	g01 *= norm[1];
	g10 *= norm[2];
	g11 *= norm[3];

	float n00 = g00.dot(cv::Vec2f(fx[0], fy[0]));
	float n10 = g10.dot(cv::Vec2f(fx[1], fy[1]));
	float n01 = g01.dot(cv::Vec2f(fx[2], fy[2]));
	float n11 = g11.dot(cv::Vec2f(fx[3], fy[3]));

	float fadex = fade(Pf[0]);
	cv::Vec2f n_x = mix(cv::Vec2f(n00, n01), cv::Vec2f(n10, n11), fadex);
	float n_xy = mix(n_x[0], n_x[1], Pf[1]);
	return 2.3*n_xy;

}
float intensity3(cv::Vec3f color){
	return color[0] + color[1] + color[2];
}

float Hue(cv::Vec3f color){
	float M = MAX(MAX(color[0], color[1]), color[2]);
	float m = MIN(MIN(color[0], color[1]), color[2]);
	float C = M - m;
	if (C == 0.0) {
		return -1.0;
	}
	float H = 0.0;
	if (M == color[2]) {
		H = mod((color[1] - color[0]) / C, 6.0);
	}
	else if (M == color[1]){
		H = (color[0] + color[2]) / C + 2.0;
	}
	else{
		H = (color[2] - color[1]) / C + 4.0;
	}
	H *= 60;
	return H;
}
float Saturation(cv::Vec3f color){
	float M = MAX(MAX(color[0], color[1]), color[2]);
	float m = MIN(MIN(color[0], color[1]), color[2]);
	if (M == 0.0) {
		return 0.0;
	}
	return (M - m) / M;
}
float Brightness(cv::Vec3f color){
	return MAX(MAX(color[0], color[1]), color[2]);
}
bool NearColorHue(cv::Vec3f color1, cv::Vec3f color2){
	float hue1 = Hue(color1);
	float hue2 = Hue(color2);
	/*if (hue1 < 0.0 || hue2 < 0.0) {
		return true;
	}*/
	if (hue1 >= hue2) {
		if (hue1 - hue2 < 10.0 || hue2 - hue1 + 360 < 10) {//30.0,30
			return true;
		}
		else{
			return false;
		}
	}
	else{
		if (hue2 - hue1 <10.0 || hue1 - hue2 + 360 < 10) {
			return true;
		}
		else{
			return false;
		}

	}
	//return false;
}
float NoiseValue(cv::Vec2f coord){
	return cnoise(coord);
}

//input:CV_32FC4,output:CV_32FC4
cv::Mat StyleTransfer::boundaryDistort(cv::Mat input4f, cv::Mat mask, cv::Mat perlinNoiseImage, int is_white, int flag){
	cv::Mat output, input;
	std::vector<cv::Mat> vec_mat;
	split(input4f, vec_mat);
	vec_mat.pop_back();
	cv::merge(vec_mat, input);//4f to 3f
	
	input4f.copyTo(output);//解决右下白边问题
	float distort_scale = 2.0;//2.0
	//float noise_scale = 0.18;//0.15
	int width = input.size().width;
	int height = input.size().height;
	int p_width = perlinNoiseImage.cols;
	int p_height = perlinNoiseImage.rows;
	unsigned int seed = 237;
	PerlinNoise pn(seed);
	
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (flag == 1 && mask.at<uchar>(row, col) < 200)
			{
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(0.0, 0.0, 0.0, 0.0);
				continue;
			}
			float row1 = ((float)row) / ((float)(height-1))*(p_height-1);
			float col1 = ((float)col) / ((float)(width - 1))*(p_width - 1);
			float row2 = ((float)col) / ((float)(width - 1))*(p_height - 1);
			float col2 = (1.0 - ((float)row) / ((float)(height - 1)))*(p_width - 1);
			float row3 = (1.0 - ((float)row) / ((float)(height - 1)))*(p_height - 1);
			float col3 = (1.0 - ((float)col) / ((float)(width-1)))*(p_width-1);
			float row4 = (1.0 - ((float)col) / ((float)(width - 1)))*(p_height - 1);
			float col4 = ((float)row) / ((float)(height - 1))*(p_width - 1);

			/*float u1 = (pn.noise(10 * col1, 10 * row1, 0.8) - 0.5) * 3.0 * distort_scale;
			float v1 = (pn.noise(10 * col2, 10 * row2, 0.8) - 0.5) * 3.0 * distort_scale;
			float u2 = (pn.noise(10 * col3, 10 * row3, 0.8) - 0.5) * 3.0 * distort_scale;
			float v2 = (pn.noise(10 * col4, 10 * row4, 0.8) - 0.5) * 3.0 * distort_scale;*/
			
			float a = (perlinNoiseImage.at<cv::Vec3b>(row1, col1)[0]);
			float b = (perlinNoiseImage.at<cv::Vec3b>(row2, col2)[0]);
			float c = (perlinNoiseImage.at<cv::Vec3b>(row3, col3)[0]);
			float d = (perlinNoiseImage.at<cv::Vec3b>(row4, col4)[0]);
			

			float u1 = ( a/ 255.0 - 0.5)*3.0*distort_scale;
			float v1 = ( b/ 255.0 - 0.5)*3.0*distort_scale;
			float u2 = ( c/ 255.0 - 0.5)*3.0*distort_scale;
			float v2 = ( d/ 255.0 - 0.5)*3.0*distort_scale;


			if (u1 + col < 0 || u1 + col >= width || v1 + row < 0 || v1 + row >= height || v2 + row < 0 || v2 + row >= height || u2 + col < 0 || u2 + col >= width) {
				continue;
			}
			cv::Vec3f color1c = input.at<cv::Vec3f>(row + v1, col + u1);
			cv::Vec3f color2c = input.at<cv::Vec3f>(row + v2, col + u2);
			cv::Vec3f colorc = input.at<cv::Vec3f>(row, col);
			cv::Vec3f color_dark, color_light;
			if (intensity3(color1c) > intensity3(color2c)) {
				color_dark = color2c;
				color_light = color1c;
			}
			else{
				color_dark = color1c;
				color_light = color2c;
			}
			if (input4f.at<cv::Vec4f>(row + v1, col + u1)[3] > 0.0 || input4f.at<cv::Vec4f>(row + v2, col + u2)[3] > 0.0) {//在湿画法区域不出现
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
			}
			//面部不出现
			else if (flag == 0 && mask.at<uchar>(row,col)>200)
			{
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
			}
			/*else if (NearColorHue(color1c, color2c)){
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
			}*/
			//else if (Saturation(color1c) < 0.15 && Saturation(color2c) < 0.15){//0.2,0.2
			//	output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
			//}
			//else if (Brightness(color1c) < 0.15 && Brightness(color2c) < 0.3){//0.15,0.3
			//	output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
			//}
			/*else if (Brightness(color2c) < 0.1 && Brightness(color1c) < 0.1){
			output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
			}*/
			else if (dist(color1c, color2c) < 0.25){//0.12
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
				
			}
			else if ((color1c == color_dark) && (color2c == color_light)){//重叠				
				float alpha2 = 0.5;
				cv::Vec3f final_color;
				if (intensity3(color1c) > intensity3(color2c)) {
					alpha2 = 1.0 - Brightness(color2c);
					cv::Vec3f mixcolor = 0.5*(color1c + color2c);
					final_color = mixcolor - 0.6 * (mixcolor - cv::Vec3f(mixcolor[0] * mixcolor[0], mixcolor[1] * mixcolor[1], mixcolor[2] * mixcolor[2]));
				}
				else{
					alpha2 = 1.0 - Brightness(color1c);
					cv::Vec3f mixcolor = 0.5*(color1c + color2c);
					final_color = mixcolor - 0.6 * (mixcolor - cv::Vec3f(mixcolor[0] * mixcolor[0], mixcolor[1] * mixcolor[1], mixcolor[2] * mixcolor[2]));
				}
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(final_color[0], final_color[1], final_color[2], 1.0);
				
			}
			else if ((color1c == color_light) && (color2c == color_dark) && ((flag==0 && mask.at<uchar>(row, col)<10)||flag==1)){//留白
				//output.at<cv::Vec4f>(row, col) = cv::Vec4f(min(color1c[0] + 0.3, 1.0), min(color1c[1] + 0.3, 1.0), min(color1c[2] + 0.3, 1.0), 1.0);
				//暂时不留白
				if (is_white>0)//留白
					output.at<cv::Vec4f>(row, col) = cv::Vec4f(0.9, 0.9, 0.9, 1.0);
			}
			else
			{
				output.at<cv::Vec4f>(row, col) = cv::Vec4f(colorc[0], colorc[1], colorc[2], 1.0);
			}

		}
	}

	return output;
}