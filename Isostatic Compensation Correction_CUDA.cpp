// Isostatic Compensation Correction_CUDA.cpp : 定义控制台应用程序的入口点。
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define _CRT_SECURE_NO_WARNINGS                 //用以解决scanf等unsafe的问题
#include<stdio.h>
#include<math.h>
#include<windows.h>
#include<time.h>
#include<omp.h>
#include<ppl.h>
#include<iostream>
#include <stdlib.h>
#include <string.h>
using namespace std;
using namespace concurrency;
#include "cublas_v2.h"

int DW68(int i, int j)              //   6*8
{
	return(i * 480 + j);
}
int DW46(int i, int j)              //   4*6
{
	return(i * 360 + j);
}


__global__ void dxgz_kernel(int numElements, double dx, double dy, int *d_H, int *row_id46, double *d_g)
{
	int tid = blockDim.x *blockIdx.x + threadIdx.x;

	int e, f;
	int n, m;
	int dh;
	double i1, i2, j1, j2;
	double i1dx, i2dx, j1dy, j2dy;
	n = row_id46[tid];
	m = tid - n * 360;
	n = n + 60;
	m = m + 60;

	//if (tid == 0) d_g[0] = 1.11;
	if (tid< numElements)
	{
		for (e = n - 60; e<n + 60; e++)
		{
			for (f = m - 60; f<m + 60; f++)
			{
				i1 = f - m + 0.5;
				i2 = f - m - 0.5;
				j1 = e - n + 0.5;
				j2 = e - n - 0.5;

				i1dx = i1*dx;
				i2dx = i2*dx;
				j1dy = j1*dy;
				j2dy = j2*dy;
				dh = d_H[e * 480 + f] - d_H[n * 480 + m];

				if (dh == 0)
				{
					d_g[tid] = d_g[tid] + 0.0;
				}
				else
				{
					d_g[tid] = d_g[tid] +
						(((i1dx*log(j1dy + sqrt(i1dx*i1dx + j1dy*j1dy + dh*dh)) + j1dy*log(i1dx + sqrt(i1dx*i1dx + j1dy*j1dy + dh*dh)) - dh*atan(i1dx*j1dy / (dh*sqrt(i1dx*i1dx + j1dy*j1dy + dh*dh))))
						- (i2dx*log(j1dy + sqrt(i2dx*i2dx + j1dy*j1dy + dh*dh)) + j1dy*log(i2dx + sqrt(i2dx*i2dx + j1dy*j1dy + dh*dh)) - dh*atan(i2dx*j1dy / (dh*sqrt(i2dx*i2dx + j1dy*j1dy + dh*dh)))))

						- ((i1dx*log(j2dy + sqrt(i1dx*i1dx + j2dy*j2dy + dh*dh)) + j2dy*log(i1dx + sqrt(i1dx*i1dx + j2dy*j2dy + dh*dh)) - dh*atan(i1dx*j2dy / (dh*sqrt(i1dx*i1dx + j2dy*j2dy + dh*dh))))
						- (i2dx*log(j2dy + sqrt(i2dx*i2dx + j2dy*j2dy + dh*dh)) + j2dy*log(i2dx + sqrt(i2dx*i2dx + j2dy*j2dy + dh*dh)) - dh*atan(i2dx*j2dy / (dh*sqrt(i2dx*i2dx + j2dy*j2dy + dh*dh))))))


						-


						(((i1dx*log(j1dy + sqrt(i1dx*i1dx + j1dy*j1dy)) + j1dy*log(i1dx + sqrt(i1dx*i1dx + j1dy*j1dy)))
						- (i2dx*log(j1dy + sqrt(i2dx*i2dx + j1dy*j1dy)) + j1dy*log(i2dx + sqrt(i2dx*i2dx + j1dy*j1dy))))

						- ((i1dx*log(j2dy + sqrt(i1dx*i1dx + j2dy*j2dy)) + j2dy*log(i1dx + sqrt(i1dx*i1dx + j2dy*j2dy)))
						- (i2dx*log(j2dy + sqrt(i2dx*i2dx + j2dy*j2dy)) + j2dy*log(i2dx + sqrt(i2dx*i2dx + j2dy*j2dy)))));
				}
			}
		}
		d_g[tid] = d_g[tid] * 2.67*6.67428*0.001 / 2;
	}

}

int main()
{
	int n, m, e, f;

	double dks, djs, dt, ks, js, t;


	//int *H;
	//H = (int*)malloc(480* 360 * sizeof(int));   memset(H, 0, 480 * 360 * sizeof(int));

	int *H;
	if (cudaHostAlloc(&H, 480 * 360 * sizeof(int), cudaHostAllocPortable) != cudaSuccess)
	{
		printf("cudaHostAlloc Failed\n");
	}

	FILE *fp;
	fp = fopen("D:\\1.data\\srtm\\H.txt", "r");

	if ((fp = fopen("D:\\1.data\\srtm\\H.txt", "r")) == NULL)
	{
		printf("Can't open the file!\n");
		exit(0);
	}

	for (n = 0; n <360; n++)
	{
		for (m = 0; m <480; m++)
		{
			fscanf(fp, "%d", &H[DW68(n, m)]);
		}
	}
	fclose(fp);

	/*for (n = 0; n <360; n++)
	{
	for (m = 0; m < 480; m++)
	{
	printf("%d\t", H[DW68(n, m)]);
	}
	printf("\n");
	}
	*/
	double *g;
	g = (double*)malloc(360 * 240 * sizeof(double));   
	memset(g, 1, 360 * 240 * sizeof(double));

	/*double *g;
	if (cudaHostAlloc(&g, 360 * 240 * sizeof(double), cudaHostAllocPortable) != cudaSuccess)
	{
	printf("cudaHostAlloc Failed\n");
	}*/

	/*for (n = 0; n <240; n++)
	{
	for (m = 0; m < 360; m++)
	{
	g[DW46(n, m)]=g[DW46(n, m)]+1;
	}
	}*/

	double Bm = 3.1415926 * 27 / 360;
	double dB = 3.1415926 * 1 / (360 * 60);
	double dL = 3.1415926 * 1 / (360 * 60);

	double dx, dy;
	double R = 6378137;
	dx = R*dB;
	dy = R*cos(Bm)*dL;

	int numElements = 360 * 240;
	//int *d_H;
	double *d_g;
	//cudaMalloc((void**)&d_H, 480 * 360 * sizeof(int)); cudaMemset(d_H, 0, 480 * 360 * sizeof(int));
	cudaMalloc((void**)&d_g, numElements * sizeof(double)); cudaMemset(d_g, 0, numElements * sizeof(double));

	int *row_id46;
	if (cudaHostAlloc(&row_id46, 360 * 240 * sizeof(int), cudaHostAllocPortable) != cudaSuccess)
	{
		printf("cudaHostAlloc Failed\n");
	}
	for (int i = 0; i < 240; i++)
	{
		for (int j = 0; j < 360; j++)                                 //确定一维数组中对应二维数组的行号
		{
			row_id46[DW46(i, j)] = i;
		}
	}

	//for (n = 0; n <240; n++)
	//{
	//for (m = 0; m < 360; m++)
	//{
	//	printf("%d\t", row_id46[DW46(n, m)]);
	//}
	//printf("\n");
	//}

	int blockperdim = int((numElements + 127) / 128);


	//检验用
	//int *Hj;
	//Hj = (int*)malloc(480 * 360 * sizeof(int));   memset(Hj, 0, 480 * 360 * sizeof(int));


	ks = clock();
	//cudaMemcpy(d_H, H, 480 * 360 * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(Hj, d_H, 480 * 360 * sizeof(int), cudaMemcpyDeviceToHost);

	//printf("hello");
	//Kernel调用出错的原因有很多，在Host端调用Kernel函数后，使用如下语句：
	cudaError_t  error_check;
	//....
	/*dxgz_kernel << <blockperdim, 128 >> >(numElements, dx, dy, H, row_id46, d_g );
	printf("aaa=%f\n", g[0]);
	error_check = cudaGetLastError();
	if (error_check != cudaSuccess)
	{
	printf("%s\n", cudaGetErrorString(error_check));
	system("pause");
	return 0;
	}*/
	dxgz_kernel << <blockperdim, 128 >> >(numElements, dx, dy, H, row_id46, d_g);

	cudaMemcpy(g, d_g, 360 * 240 * sizeof(double), cudaMemcpyDeviceToHost);
	js = clock();

	printf("\n");

	t = (js - ks) / 1000;//2855.32593秒 47.58876分钟
	printf("计算耗时：%8.5f\t秒", t);
	printf("\n");

	/*t = t / 60;
	printf("%8.5f\t", t);

	printf("%f\t", g[DW46(200, 200)]);*/

	/*FILE *dxgzCX;
	dxgzCX = fopen("F:\\01、程序\\2018.11.07、地形改正\\dxgzCUDA\\g.txt", "w");
	for (n = 0; n < 240; n++)
	{
	for (m = 0; m < 360; m++)
	{
	fprintf(dxgzCX, "%f\t", n, m, g[DW46(n, m)]);
	}
	fprintf(dxgzCX, "\n");
	}
	fclose(dxgzCX);*/

	//free(H);
	cudaFreeHost(H);
	free(g);
	cudaFreeHost(row_id46);
	//cudaFree(d_H);
	//cudaFree(d_g);
	getchar();
	return 0;
}