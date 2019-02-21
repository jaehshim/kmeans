#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "kmeans.h"
#include <float.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

char *get_source_code(const char *file_name, size_t *len) {
        char *source_code;
        size_t length;
        FILE *file = fopen(file_name, "r");
        if (file == NULL) {
                printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
                exit(EXIT_FAILURE);
        }

        fseek(file, 0, SEEK_END);
        length = (size_t)ftell(file);
        rewind(file);

        source_code = (char *)malloc(length + 1);
        fread(source_code, length, 1, file);
        source_code[length] = '\0';

        fclose(file);

        *len = length;
        return source_code;
}
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue[2];
cl_program program;
char* kernel_source;
size_t kernel_source_size;
cl_kernel kernel;
cl_kernel kernel2;
cl_int err;

void kmeans_init() {
	err = clGetPlatformIDs(1, &platform, NULL);
        CHECK_ERROR(err);

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        CHECK_ERROR(err);

        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        CHECK_ERROR(err);

        queue[0] = clCreateCommandQueue(context, device, 0, &err);
        CHECK_ERROR(err);
        queue[1] = clCreateCommandQueue(context, device, 0, &err);
        CHECK_ERROR(err);

        kernel_source = get_source_code("kernel.cl", &kernel_source_size);
        program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
        CHECK_ERROR(err);

        err = clBuildProgram(program, 1, &device, "", NULL, NULL);
        if (err == CL_BUILD_PROGRAM_FAILURE)  {
                size_t log_size;
                char * log;

                err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                CHECK_ERROR(err);

                log = (char *)malloc(log_size+1);
                err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                CHECK_ERROR(err);

                log[log_size] = '\0';
                printf("Compiler Error:\n%s\n", log);
                free(log);
                exit(0);
        }
        CHECK_ERROR(err);

        kernel = clCreateKernel(program, "kmeans", &err);
        CHECK_ERROR(err);
        kernel2 = clCreateKernel(program, "kmeans_2", &err);
        CHECK_ERROR(err);
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
	int i, data_i, class_i;
	size_t global_size = data_n/2;
	size_t local_size = 1024;
	int* count = (int*)malloc(sizeof(int) * class_n);

	printf("class_n : %d\ndata_n : %d\n", class_n, data_n);

	cl_mem bufData[2], bufCent, bufPart[2], bufCount;
	cl_event kernel_event[2];
	
	bufData[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*data_n, NULL, &err);
	CHECK_ERROR(err);
	bufData[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*data_n, NULL, &err);
	CHECK_ERROR(err);
	bufCent = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*2*class_n, NULL, &err);
	CHECK_ERROR(err);	
	bufPart[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*(data_n/2), NULL, &err);
	CHECK_ERROR(err);	
	bufPart[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*(data_n/2), NULL, &err);
	CHECK_ERROR(err);	
	bufCount = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*class_n, NULL, &err);
	CHECK_ERROR(err);


	err = clEnqueueWriteBuffer(queue[0], bufData[0], CL_FALSE, 0, sizeof(float)*data_n, (float*)data, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue[0], bufData[1], CL_FALSE, 0, sizeof(float)*data_n, (float*)(data+data_n/2), 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &bufCent);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &bufCount);
	CHECK_ERROR(err);
//	err = clSetKernelArg(kernel2, 2, sizeof(int)*class_n, NULL);
//	CHECK_ERROR(err);

	for (i = 0; i < iteration_n; i++) {
		global_size = data_n/2;
		local_size = 1024;
		
		err = clEnqueueWriteBuffer(queue[0], bufCent, CL_FALSE, 0, sizeof(float)*2*class_n, (float*)centroids, 0, NULL, NULL);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufData[0]);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufCent);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufPart[0]);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 3, sizeof(int), &class_n);
		CHECK_ERROR(err);
		
		err = clEnqueueNDRangeKernel(queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event[0]);
		CHECK_ERROR(err);
		err = clFlush(queue[0]);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufData[1]);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufCent);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufPart[1]);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 3, sizeof(int), &class_n);
		CHECK_ERROR(err);
		
		err = clEnqueueNDRangeKernel(queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event[1]);
		CHECK_ERROR(err);
                err = clFlush(queue[0]);
                CHECK_ERROR(err);

		err = clEnqueueReadBuffer(queue[1], bufPart[0], CL_TRUE, 0, sizeof(int)*(data_n/2), partitioned, 1, &kernel_event[0], NULL);
		CHECK_ERROR(err);
		err = clFlush(queue[1]);
		CHECK_ERROR(err);

		err = clEnqueueReadBuffer(queue[1], bufPart[1], CL_TRUE, 0, sizeof(int)*(data_n/2), partitioned+(data_n)/2, 1, &kernel_event[1], NULL);
		CHECK_ERROR(err);
		err = clFlush(queue[1]);
		CHECK_ERROR(err);

		err = clFinish(queue[0]);
		CHECK_ERROR(err);
		err = clFinish(queue[1]);
		CHECK_ERROR(err);

		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[class_i].x = 0.0;
			centroids[class_i].y = 0.0;
			count[class_i] = 0;
		}

		for (data_i = 0; data_i < data_n; data_i++) {         
			centroids[partitioned[data_i]].x += data[data_i].x;
			centroids[partitioned[data_i]].y += data[data_i].y;
			count[partitioned[data_i]]++;
		}

		err = clEnqueueWriteBuffer(queue[0], bufCent, CL_FALSE, 0, sizeof(float)*2*class_n, (float*)centroids, 0, NULL, NULL);
		CHECK_ERROR(err);
		err = clEnqueueWriteBuffer(queue[0], bufCount, CL_FALSE, 0, sizeof(int)*class_n, count, 0, NULL, NULL);
		CHECK_ERROR(err);

		global_size = class_n;
		if (global_size < 1024)
			local_size = class_n;
		else
			local_size = 1024;
		err = clEnqueueNDRangeKernel(queue[0], kernel2, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
		CHECK_ERROR(err);

		err = clEnqueueReadBuffer(queue[0], bufCent, CL_TRUE, 0, sizeof(float)*2*class_n, (float*)centroids, 0, NULL, NULL);
		CHECK_ERROR(err);

/*		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[class_i].x /= count[class_i];
			centroids[class_i].y /= count[class_i];
		}
*/
	}
	printf("Kmeans OpenCL Version...\n");
}
