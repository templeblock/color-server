#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "commerHeader.h"
#include "device_types.h"
#include "host_defines.h"
#include <iostream>

inline void checkCudaErrors(cudaError err)//错误处理函数
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error: %s.\n", cudaGetErrorString(err));
		return;
	}
}

__device__ float near_color(float3 color1, float3 color2){
	//BGR to YUV
	float Y1 = 0.299*color1.z + 0.587*color1.y + 0.114*color1.x;
	float U1 = -0.147*color1.z - 0.289*color1.y + 0.436*color1.x;
	float V1 = 0.615*color1.z - 0.515*color1.y - 0.100*color1.x;

	float Y2 = 0.299*color2.z + 0.587*color2.y + 0.114*color2.x;
	float U2 = -0.147*color2.z - 0.289*color2.y + 0.436*color2.x;
	float V2 = 0.615*color2.z - 0.515*color2.y - 0.100*color2.x;

	float distance = (U1 - U2)*(U1 - U2) + (V1 - V2)*(V1 - V2) + (Y1 - Y2)*(Y1 - Y2);
	return distance;
}

//mask:0,not compute
//mask:255,compute
//flag:1, use mask
__global__ void near_color_edge_detect(float3* dataIn, float4 *dataOut, int width, int height, uchar* mask, int flag)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	float3 colorC, colorL, colorR, colorT, colorB;
	float nearcolor_value = 0.01f;
	float4 out;

	if ((flag == 1 && mask[yIndex *width + xIndex] > 200) || flag == 0)
	{
		xIndex = MIN(xIndex + 1, width - 1);

		colorC = dataIn[yIndex *width + xIndex ];
		colorL = dataIn[yIndex *width + MAX(xIndex - 1,0)];
		colorR = dataIn[yIndex *width + MIN(xIndex + 1, width - 1)];
		colorT = dataIn[MAX(yIndex - 1, 0) *width + xIndex];
		colorB = dataIn[MIN(yIndex + 1, height-1) *width + xIndex];
		
		out.x = 0.0f;
		out.y = 0.0f;
		out.z = 0.0f;
		out.w = 0.0f;

		if (near_color(colorR, colorL) > nearcolor_value) {
			if (near_color(colorL, colorC) < 
				near_color(colorR, colorC)) {
				out.x = 1.0f;				
				out.y = 0.0f;
				out.z = 0.0f;
				out.w = 1.0f;
			}
			else{
				out.x = -1.0f;
				out.y = 0.0f;
				out.z = 0.0f;
				out.w = 1.0f;
			}
		}
		if (near_color(colorT, colorB) > nearcolor_value) {
			if (near_color(colorT, colorC) < 
				near_color(colorC, colorB)) {
				out.x = 0.0f;
				out.y = 1.0f;
				out.z = 0.0f;
				out.w = 1.0f;
			}
			else{
				out.x = 0.0f;
				out.y = -1.0f;
				out.z = 0.0f;
				out.w = 1.0f;
			}
		}		
	}
	else if (flag == 1 && mask[yIndex *width + xIndex] < 200){
		out.x = 0.0f;
		out.y = 0.0f;
		out.z = 0.0f;
		out.w = 0.0f;
		
	}
	dataOut[yIndex *width + xIndex] = out;
}

__global__ void median_blur(float3* dataIn, float3 *dataOut, int kernel, int width, int height, uchar* mask, int flag)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	float3 u;
	u.x = 0.0;
	u.y = 0.0;
	u.z = 0.0;
	int y, x;

	if ((flag == 1 && mask[yIndex *width + xIndex] > 200) || flag == 0)
	{
		for (int i = -kernel; i < kernel; i++)
		{
			for (int j = -kernel; j < kernel; j++)
			{
				y = MIN(MAX(0, yIndex + j), width - 1);
				x = MIN(MAX(0, xIndex + i), height - 1);
				
				u.x += dataIn[y *width + x].x;
				u.y += dataIn[y *width + x].y;
				u.z += dataIn[y *width + x].z;
			}
		}
		dataOut[yIndex *width + xIndex].x = u.x / ((kernel * 2 )*(kernel * 2 ));
		dataOut[yIndex *width + xIndex].y = u.y / ((kernel * 2 )*(kernel * 2 ));
		dataOut[yIndex *width + xIndex].z = u.z / ((kernel * 2 )*(kernel * 2 ));
	}
	else if (flag == 1 && mask[yIndex *width + xIndex] < 200)
	{
		dataOut[yIndex *width + xIndex] = dataIn[yIndex *width + xIndex];
	}
}

__global__ void large_gradient(float3* dataIn, float3 *dataOut, int width, int height, uchar* mask, int flag)
{

	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	int kernel = 3;
	float3 u, v;

	if (xIndex + kernel < width && yIndex + kernel < height &&xIndex - kernel >= 0 && yIndex - kernel >= 0)
	{
		if ((flag == 1 && mask[yIndex *width + xIndex] > 200) || flag==0){
			u.x = 0.0; u.y = 0.0; u.z = 0.0;
			v.x = 0.0; v.y = 0.0; v.z = 0.0;
			for (int i = 1; i <= kernel; i++) {
				for (int j = -kernel; j <= kernel; j++) {
					u.x += dataIn[(yIndex + j) *width + (xIndex + i)].x - dataIn[(yIndex + j) *width + (xIndex - i)].x;
					u.y += dataIn[(yIndex + j) *width + (xIndex + i)].y - dataIn[(yIndex + j) *width + (xIndex - i)].y;
					u.z += dataIn[(yIndex + j) *width + (xIndex + i)].z - dataIn[(yIndex + j) *width + (xIndex - i)].z;
					v.x += dataIn[(yIndex + i) *width + (xIndex + j)].x - dataIn[(yIndex - i) *width + (xIndex + j)].x;
					v.y += dataIn[(yIndex + i) *width + (xIndex + j)].y - dataIn[(yIndex - i) *width + (xIndex + j)].y;
					v.z += dataIn[(yIndex + i) *width + (xIndex + j)].z - dataIn[(yIndex - i) *width + (xIndex + j)].z;
				}
			}
			float norm = (kernel * (kernel * 2 + 1))*0.4;
			v.x /= norm; v.y /= norm; v.z /= norm;
			u.x /= norm; u.y /= norm; u.z /= norm;

			dataOut[yIndex *width + xIndex].x = (u.x + u.y + u.z) / 3.0;
			dataOut[yIndex *width + xIndex].y = (v.x + v.y + v.z) / 3.0;
			dataOut[yIndex *width + xIndex].z = 0.0;
		}
		else if (flag == 1 && mask[yIndex *width + xIndex] < 200){
			dataOut[yIndex *width + xIndex].x = 0.0;
			dataOut[yIndex *width + xIndex].y = 0.0;
		}
	}
}
__device__ float3 get_hls(float3 color){
	int H, L, S;
	float M = MAX(MAX(color.x, color.y), color.z);
	float m = MIN(MIN(color.x, color.y), color.z);
	float C = M - m;
	if (C == 0.0) {
		H = -1.0;
	}
	else if (M == color.z) {
		float t = color.y - color.x / C;
		H = t - (int)t/6*6;
	}
	else if (M == color.y){
		H = (color.x - color.z) / C + 2.0;
	}
	else{
		H = (color.z - color.y) / C + 4.0;
	}
	H *= 60.0;
	L = 0.5 * (M + m);
	if (C == 0.0) {
		S = 0.0;
	}
	else{
		S = C / (1.0 - fabs(2.0 * L - 1.0));
	}
	float3 output;
	output.x = H;
	output.y = L;
	output.z = S;
	return output;
}

__device__ float hue_distance(float hue1, float hue2){
	if (hue1 >= hue2)
		return MIN(hue1 - hue2, 360.0 - hue1 + hue2);
	else
		return MIN(hue2 - hue1, 360.0 - hue2 + hue1);
}

//flag:1,error_mask
//flag:0,face_mask
__global__ void edge_type_detect(float3* dataIn, float4 *near_color,float3 *grad,uchar* mask,float4* output, int width, int height, int flag)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	float3 hlsimage, hlsedge;
	float gradient_x, gradient_y;
	bool nearhue;
	int drow, dcol;
	float3 near2;

	if ((flag == 1 && mask[yIndex *width + xIndex] > 200) || flag == 0){
		drow = (int)near_color[yIndex *width + xIndex].y;
		dcol = (int)near_color[yIndex *width + xIndex].x;

		near2.x = dataIn[(yIndex + drow) *width + (xIndex + dcol)].x;
		near2.y = dataIn[(yIndex + drow) *width + (xIndex + dcol)].y;
		near2.z = dataIn[(yIndex + drow) *width + (xIndex + dcol)].z;

		output[yIndex *width + xIndex].x = near2.x;
		output[yIndex *width + xIndex].y = near2.y;
		output[yIndex *width + xIndex].z = near2.z;

		if (fabs(near_color[yIndex *width + xIndex].w - 1.0) < 0.000001)
		{
			hlsimage = get_hls(dataIn[yIndex *width + xIndex]);
			hlsedge = get_hls(near2);
			gradient_x = grad[yIndex *width + xIndex].x;
			gradient_y = grad[yIndex *width + xIndex].y;
			nearhue = hue_distance(hlsedge.x, hlsimage.x) < 90.0;
			output[yIndex *width + xIndex].w = 1.0;
			if (flag == 0 && mask[yIndex *width + xIndex] > 100)//面部
			{
				output[yIndex *width + xIndex].w = 1.0;
			}
			else if (fabs(gradient_x) < 0.1 && fabs(gradient_y) < 0.1) {
				output[yIndex *width + xIndex].w = 1.0;//不添加湿画区域
			}
			else if (hue_distance(hlsedge.x, hlsimage.x) < 20.0){
				output[yIndex *width + xIndex].w = 0.5;//湿画区域
			}
			else if (hue_distance(hlsedge.x, hlsimage.x) < 90.0&&fabs(hlsimage.y - hlsedge.y) < 0.2)
			{
				output[yIndex *width + xIndex].w = 0.5;
			}
			else if ((hlsimage.z > 0.3 && hlsedge.z > 0.3 && !nearhue) || fabs(hlsimage.y - hlsedge.y) > 0.6 || (fabs(gradient_x) < 0.1 && fabs(gradient_y) < 0.1)){
				output[yIndex *width + xIndex].w = 1.0;
			}
			else{
				output[yIndex *width + xIndex].w = 0.0;
			}
		}
	}
}
__global__ void wet_pre(float3* dataIn, float4 *edge_type, float3 *grad, uchar* random_noise, float4* output, int width, int height, int r_width, int r_height, uchar* mask, int flag)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int half_width = 6;
	float noise_control = 0.4;
	float3 color, other_color;
	float in_out = 0.0, dist = 0.0;
	float nearestdist = 100.0;

	if ((flag == 1 && mask[row *width + col] > 200) || flag == 0){
		color = dataIn[row * width + col];
		for (int i = -half_width; i <= half_width; i++) {
			for (int j = -half_width; j < half_width; j++) {
				if (row + i < 0 || row + i >= height || col + j < 0 || col + j >= width) {
					continue;
				}
				if (fabs(edge_type[(row + i) * width + col + j].w - 0.5) < 0.00001){
					dist = sqrtf(i*i + j*j);
					if (dist < nearestdist) {
						nearestdist = dist;
						in_out = 1.0;
						other_color.x = edge_type[(row + i) * width + col + j].x;
						other_color.y = edge_type[(row + i) * width + col + j].y;
						other_color.z = edge_type[(row + i) * width + col + j].z;
					}
				}
			}
		}
		nearestdist /= sqrtf(2.0) * half_width;
		if (in_out == 1.0){
			if (0.299*color.z + 0.587*color.y + 0.114*color.x < 0.299*other_color.z + 0.587*other_color.y + 0.114*other_color.x)
				in_out = 0.5;
		}
		output[row * width + col].x = color.x;
		output[row * width + col].y = color.y;
		output[row * width + col].z = color.z;
		output[row * width + col].w = in_out;

		if (fabs(in_out - 1.0) < 0.00001){//湿画区域
			int rown = (int)((float)(row)* r_height / ((float)(height)));
			int coln = (int)((float)(col)* r_width / ((float)(width)));
			uchar colornoise = random_noise[rown * r_width + coln];
			//uchar colornoise = 200;
			float noise = (float)(colornoise) / (255.0);
			if (noise * (1 - nearestdist) * 2 > noise_control * 2) {
				output[row * width + col].x = other_color.x;
				output[row * width + col].y = other_color.y;
				output[row * width + col].z = other_color.z;
				output[row * width + col].w = in_out;
			}
		}
	}
}

__device__ float WeightCal(float x, float y, float3 theta, float maxA, float maxB){
	float y2 = -theta.y * x + theta.x * y;
	float x2 = theta.x * x + theta.y * y;
	return MAX(1.0 - (x2 * x2 / maxA + y2 * y2 / maxB), 0.0);
}

__global__ void wet_filter(float4* wet_pre, float3 *grad, float4* output, int width, int height, int level, uchar* mask, int flag)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	//int wet_degree = 15;//原为10
	int wet_degree = level;//原为10
	int maxR = int(ceil(float(wet_degree) / 3.0));//大于等于的最小整数,3
	float maxA = float((maxR + 1) * (maxR + 1));//半长轴，16
	float k = wet_degree / 40.0;//原来为80
	k = MIN(MAX(0.25, k), 1.0);
	float maxB = maxA * (k*k);//半短轴，1
	float weightSum = 1.0;
	int weight_inout = 0;

	output[row * width + col] = wet_pre[row * width + col];
	float3 direction = grad[row * width + col];

	if ((flag == 1 && mask[row *width + col] > 200) || flag == 0){
		if (fabs(direction.x) < 0.000001&&fabs(direction.y) < 0.000001)
		{

		}
		else if (fabs(direction.x) > fabs(direction.y))
		{
			direction.y = direction.y / fabs(direction.x);
			if (direction.x < 0)
				direction.x = -1.0;
			else
				direction.x = 1.0;
		}
		else
		{
			direction.x = direction.x / fabs(direction.y);
			if (direction.y < 0)
				direction.y = -1.0;
			else
				direction.y = 1.0;
		}
		float4 cur_color = wet_pre[row * width + col];
		float3 colorSum;
		colorSum.x = cur_color.x;
		colorSum.y = cur_color.y;
		colorSum.z = cur_color.z;

		if (cur_color.w >= 1.0) {//wet zone
			for (int i = 1; i <= maxR; i++) {
				float weight = WeightCal(0.0, float(i), direction, maxA, maxB);
				int row2 = row + i;
				if (row2 < height && row2 >= 0) {
					float4 color = wet_pre[row2 * width + col];
					colorSum.x += weight*color.x;
					colorSum.y += weight*color.y;
					colorSum.z += weight*color.z;
					weightSum += weight;
					if (fabs(color.w - 1.0) < 0.000001) {
						weight_inout++;
					}
				}
				row2 = row - i;
				if (row2 < height && row2 >= 0) {
					float4 color = wet_pre[row2 * width + col];
					colorSum.x += weight*color.x;
					colorSum.y += weight*color.y;
					colorSum.z += weight*color.z;
					weightSum += weight;
					if (fabs(color.w - 1.0) < 0.000001) {
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
						float4 color = wet_pre[row2 * width + col2];
						colorSum.x += weight*color.x;
						colorSum.y += weight*color.y;
						colorSum.z += weight*color.z;
						weightSum += weight;
						if (fabs(color.w - 1.0) < 0.000001) {
							weight_inout++;
						}
					}
					row2 = row - j;
					col2 = col - i;
					if (row2 < height && row2 >= 0 && col2 < width && col2 >= 0) {
						float4 color = wet_pre[row2 * width + col2];
						colorSum.x += weight*color.x;
						colorSum.y += weight*color.y;
						colorSum.z += weight*color.z;
						weightSum += weight;
						if (fabs(color.w - 1.0) < 0.000001) {
							weight_inout++;
						}
					}
				}
			}
			colorSum.x /= weightSum;
			colorSum.y /= weightSum;
			colorSum.z /= weightSum;
			if (weight_inout > 0) {
				output[row * width + col].x = colorSum.x;
				output[row * width + col].y = colorSum.y;
				output[row * width + col].z = colorSum.z;
				output[row * width + col].w = cur_color.w;
			}
			else{
				output[row * width + col] = cur_color;
			}
		}
	}
			
}
//0:face_mask, image
//1:error_mask video
extern "C" void wet_main_cuda(cv::Mat abst, cv::Mat& dst, cv::Mat random_mat, int level, cv::Mat mask, int flag)
{
	int width = abst.cols;
	int height = abst.rows;
	int r_width = random_mat.cols;
	int r_height = random_mat.rows;

	float3 *d_in, *blur, *grad;
	float4* near_color, *edge_type, *wetpre, *wet;
	uchar* cmask, *random_noise;
	//cv::Mat mask = cv::Mat::zeros(abst.size(), CV_8U);

	checkCudaErrors(cudaMalloc((void**)&d_in, sizeof(float3)* width*height));
	checkCudaErrors(cudaMalloc((void**)&near_color, sizeof(float4)* width*height));
	checkCudaErrors(cudaMalloc((void**)&blur, sizeof(float3)* width*height));
	checkCudaErrors(cudaMalloc((void**)&edge_type, sizeof(float4)* width*height));
	checkCudaErrors(cudaMalloc((void**)&grad, sizeof(float3)* width*height));
	checkCudaErrors(cudaMalloc((void**)&wetpre, sizeof(float4)* width*height));
	checkCudaErrors(cudaMalloc((void**)&wet, sizeof(float4)* width*height));
	checkCudaErrors(cudaMalloc((void**)&cmask, sizeof(uchar)* width*height));
	checkCudaErrors(cudaMalloc((void**)&random_noise, sizeof(uchar)*r_width *r_height));

	checkCudaErrors(cudaMemcpy(d_in, abst.data, sizeof(float3)* width*height, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(cmask, mask.data, sizeof(uchar)* width*height, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(random_noise, random_mat.data, sizeof(uchar)* r_width*r_height, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);


	near_color_edge_detect << <blocksPerGrid, threadsPerBlock >> >(d_in, near_color,width,height,cmask, flag);
	median_blur << <blocksPerGrid, threadsPerBlock >> >(d_in, blur, 9, width, height, cmask, flag);
	large_gradient << <blocksPerGrid, threadsPerBlock >> >(blur, grad, width, height, cmask, flag);
	edge_type_detect << <blocksPerGrid, threadsPerBlock >> >(d_in, near_color, grad, cmask, edge_type, width, height, flag);
	wet_pre << <blocksPerGrid, threadsPerBlock >> >(d_in, edge_type, grad, random_noise, wetpre, width, height, r_width, r_height, cmask, flag);
	wet_filter << <blocksPerGrid, threadsPerBlock >> >(wetpre, grad, wet, width, height, level, cmask, flag);

	checkCudaErrors(cudaMemcpy(dst.data, wet, width*height * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_in));
	checkCudaErrors(cudaFree(blur));
	checkCudaErrors(cudaFree(near_color));
	checkCudaErrors(cudaFree(grad));
	checkCudaErrors(cudaFree(edge_type));
	checkCudaErrors(cudaFree(cmask));
	checkCudaErrors(cudaFree(random_noise));
	checkCudaErrors(cudaFree(wetpre));
	checkCudaErrors(cudaFree(wet));
	

}