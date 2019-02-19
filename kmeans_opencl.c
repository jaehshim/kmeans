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
cl_command_queue queue;
cl_program program;
char* kernel_source;
size_t kernel_source_size;
cl_kernel kernel;
cl_int err;
cl_mem bufData, bufCent, bufPart;

void kmeans_init() {
	err = clGetPlatformIDs(1, &platform, NULL);
        CHECK_ERROR(err);

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        CHECK_ERROR(err);

        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        CHECK_ERROR(err);

        queue = clCreateCommandQueue(context, device, 0, &err);
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
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
	// Loop indices for iteration, data and class
	int i, data_i, class_i;
	// Count number of data in each class
	int* count = (int*)malloc(sizeof(int) * class_n);
	// Temporal point value to calculate distance

	bufData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*2*data_n, NULL, &err);
        CHECK_ERROR(err);
        bufCent = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*2*class_n, NULL, &err);
	CHECK_ERROR(err);	
        bufPart = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*data_n, NULL, &err);
	CHECK_ERROR(err);	


/*	err = clEnqueueWriteBuffer(queue, bufData, CL_FALSE, 0, sizeof(float)*2*data_n, (float*)data, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufCent, CL_FALSE, 0, sizeof(float)*2*class_n, (float*)centroids, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufPart, CL_FALSE, 0, sizeof(int)*data_n, partitioned, 0, NULL, NULL);
	CHECK_ERROR(err);
*/
	// Iterate through number of interations
	for (i = 0; i < iteration_n; i++) {

		err = clEnqueueWriteBuffer(queue, bufData, CL_FALSE, 0, sizeof(float)*2*data_n, (float*)data, 0, NULL, NULL);
		CHECK_ERROR(err);
		err = clEnqueueWriteBuffer(queue, bufCent, CL_FALSE, 0, sizeof(float)*2*class_n, (float*)centroids, 0, NULL, NULL);
		CHECK_ERROR(err);
		err = clEnqueueWriteBuffer(queue, bufPart, CL_FALSE, 0, sizeof(int)*data_n, partitioned, 0, NULL, NULL);
		CHECK_ERROR(err);

		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufData);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufCent);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufPart);
		CHECK_ERROR(err);
		err = clSetKernelArg(kernel, 3, sizeof(int), &class_n);
		CHECK_ERROR(err);

		// Assignment step
		size_t global_size = data_n;
		size_t local_size = 1;

		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
		CHECK_ERROR(err);

		err = clEnqueueReadBuffer(queue, bufPart, CL_TRUE, 0, sizeof(int)*data_n, partitioned, 0, NULL, NULL);
		CHECK_ERROR(err);

		// Update step
		// Clear sum buffer and class count
		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[class_i].x = 0.0;
			centroids[class_i].y = 0.0;
			count[class_i] = 0;
		}

		// Sum up and count data for each class
		for (data_i = 0; data_i < data_n; data_i++) {         
			centroids[partitioned[data_i]].x += data[data_i].x;
			centroids[partitioned[data_i]].y += data[data_i].y;
			count[partitioned[data_i]]++;
		}

		// Divide the sum with number of class for mean point
		for (class_i = 0; class_i < class_n; class_i++) {
			centroids[class_i].x /= count[class_i];
			centroids[class_i].y /= count[class_i];
		}
	}
	printf("Kmeans OpenCL Version...\n");
}
