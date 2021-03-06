
#include <stdio.h>
#include <stdlib.h>

__global__
void kernel1(int* d_data) {
	// One Thread handling one operation. 
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;	// blockDim.x * blockIdx.x // threadIdx.x -> current Thread in Block
	d_data[tid] += 1;
}


__global__
void kernel2(int* d_data, const int numElement) {
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	const int nthread = blockDim.x*gridDim.x;
	const int numElementPerThread = numElement/nthread;
	const int start = tid*numElementPerThread;
	int end = start + numElementPerThread;
	
	for(int i = start; i < end; i++) {
		d_data[i] += 1;
	}
}

// Roundrobin Kernel Computation.
__global__
void kernel2_opt(int* d_data, const int numElement) {
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	const int nthread = blockDim.x*gridDim.x;

	for(int i = tid; i < numElement; i += nthread) {	
		d_data[i] += 1;
	}
}


void demo1() {

	const int numElement = 512*1024;	
	int* h_data = (int*)malloc(sizeof(int)*numElement);
	int* gold = (int*)malloc(sizeof(int)*numElement);
	for(int i = 0; i < numElement; i++) {
		h_data[i] = rand(); // 랜덤값 부여
		gold[i] = h_data[i] + 1;	// 랜덤값에 하나 더한걸 gold라고 한다.
	}

	int* d_data;
	cudaMalloc(&d_data, sizeof(int)*numElement);	// Device Memory. malloc in CPU
	cudaMemcpy(d_data, h_data, sizeof(int)*numElement,  // h_data 가 source, d_data가 destination
		   cudaMemcpyHostToDevice);		// Device 로 메모리 이동해줌. 

	cudaEvent_t start, stop;
	cudaEventCreate(&start);	// start 라는 이벤트 생성
	cudaEventCreate(&stop);		// stop 이라는 이벤트 생성
	cudaEventRecord(start, 0);	// 두번째 파라미터는 stream 넘버
	kernel1<<<1024, 512>>>(d_data); 	//하나씩 다 더하는 작업 1024개의 블럭, 512개의 Thread per Block
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);	// CUDA에서 기록하는 방법
	printf("Kernel elapsed time: %.3f ms\n", elapsedTime);


	printf("kernel1: %s\n", cudaGetErrorString(cudaGetLastError()));


	cudaMemcpy(h_data, d_data, sizeof(int)*numElement, 	// d_data는 GPU에있음
		   cudaMemcpyDeviceToHost);				// 다시 GPU에서 가져옴
	cudaFree(d_data);			// GPU 메모리 해제

	for(int i = 0; i < numElement; i++) {
		if(h_data[i] != gold[i]) {
			printf("!!!ERROR, TEST FAILED.\n");
			return;
		}
	}
	printf("Test pass...\n");

	free(h_data);
	free(gold);
	
}


void demo2(const int numElement) {

	printf("numElement = %d\n", numElement);

	int* h_data = (int*)malloc(sizeof(int)*numElement);
	int* gold = (int*)malloc(sizeof(int)*numElement);
	for(int i = 0; i < numElement; i++) {
		h_data[i] = rand();
		gold[i] = h_data[i];
	}



	int* d_data;
	cudaMalloc(&d_data, sizeof(int)*numElement);
	cudaMemcpy(d_data, h_data, sizeof(int)*numElement, 
		   cudaMemcpyHostToDevice);
	float elapsedTime = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*CPU*/
	elapsedTime = 0.0f;
	cudaEventRecord(start, 0);
	for(int i = 0; i < numElement; i++) {
		gold[i] += 1;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("CPU elapsed time: %.3f ms\n", elapsedTime);



	/*GPU method 1*/
	elapsedTime = 0.0f;
	cudaEventRecord(start, 0);
	kernel2<<<1024, 512>>>(d_data, numElement);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("kernel2 elapsed time: %.3f ms\n", elapsedTime);
	printf("kernel2: %s\n", cudaGetErrorString(cudaGetLastError()));


	/*GPU method 2*/
	cudaMemcpy(d_data, h_data, sizeof(int)*numElement, 
		   cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	kernel2_opt<<<1024, 512>>>(d_data, numElement);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("kernel2_opt elapsed time: %.3f ms\n", elapsedTime);
	printf("kernel2: %s\n", cudaGetErrorString(cudaGetLastError()));
	

	cudaMemcpy(h_data, d_data, sizeof(int)*numElement, 
		   cudaMemcpyDeviceToHost);
	cudaFree(d_data);

	for(int i = 0; i < numElement; i++) {
		if(h_data[i] != gold[i]) {
			printf("!!!ERROR, TEST FAILED. i = %d: %d, %d\n", 
				i, h_data[i], gold[i]);
			return;
		}
	}
	printf("Test pass...\n");

	free(h_data);
	free(gold);
	
}




int main() {
	int numElement = 1*1024*1024;
	demo2(numElement); //execute once to warm up for performance measurement
	printf("\n\nstart ............................................\n");
	printf("demo2 started!\n");
	for(int i = numElement; i <= 32*1024*1024; i*=2) {
		demo2(i);
		printf("\n");
	}
	printf("demo1 started!\n");
	demo1();
	return EXIT_SUCCESS;
}

