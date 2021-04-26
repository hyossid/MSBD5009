/* 
 * COMPILE: nvcc -std=c++11 clustering_cuda_skeleton.cu clustering_impl.cpp main.cpp -o cuda
 * RUN:     ./cuda <path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>
 */

#include <iostream>
#include "clustering.h"

// Define variables or functions here
__device__ __constant__ int g_num_vs;
__device__ __constant__ int g_num_es;
__device__ __constant__ int g_epsilon;
__device__ __constant__ int g_mu;
__device__ __constant__ int *g_nbr_offs;
__device__ __constant__ int *g_nbrs;
__global__ int **sim_nbrs;

__device__
int get_num_com_nbrs(int *nbrs, int left_start, int left_end, int right_start, int right_end) {
    int left_pos = left_start, right_pos = right_start, num_com_nbrs = 0;

    while (left_pos < left_end && right_pos < right_end) {
        if (nbrs[left_pos] == nbrs[right_pos]) {
            num_com_nbrs++;
            left_pos++;
            right_pos++;
        } else if (nbrs[left_pos] < nbrs[right_pos]) {
            left_pos++;
        } else {
            right_pos++;
        }
    }
    return num_com_nbrs;
}

__device__
void expansion(int cur_id, int num_clusters, int *num_sim_nbrs, int **sim_nbrs,
               bool *visited, bool *pivots, int *cluster_result) {
    for (int i = 0; i < num_sim_nbrs[cur_id]; i++) {
        int nbr_id = sim_nbrs[cur_id][i];
        if ((pivots[nbr_id])&&(!visited[nbr_id])){
            visited[nbr_id] = true;
            cluster_result[nbr_id] = num_clusters;
            expansion(nbr_id, num_clusters, num_sim_nbrs, sim_nbrs, visited, pivots,
                        cluster_result);
        }
    }
}


__global__
void kernel(int* d_cluster_result,const int numVertex){

    const int tid = blockDim.x*blockIdx.x + threadIdx.x; // Thread id 
	const int nthread = blockDim.x*gridDim.x;   // Total number of thread

    for(int i =tid; i<numVertex; i+=nthread){
        // i 번째 vertex 를 처리
        int left_start = g_nbr_offs[i];
        int left_end = g_nbr_offs[i + 1];
        int left_size = left_end - left_start;

    }


}


__global__
void step1kernel(bool* d_pivots,int* d_num_sim_nbrs,int num_vs,int* nbrs,int* nbr_offs){

    const int tid = blockDim.x*blockIdx.x + threadIdx.x; // Thread id 
	const int nthread = blockDim.x*gridDim.x;   // Total number of thread

    for (int i =tid; i<num_vs; i+=nthread) {
        int *left_start = &nbrs[nbr_offs[i]];
        int *left_end = &nbrs[nbr_offs[i + 1]]; //left_end는 다음 노드의 첫 이웃이니까 계산하면 안됨
        int left_size = left_end - left_start;  // 배열 차이 사이즈를 담고있겠지. (nbrs에서 다음노드의 첫이웃 - 현재노드의 첫 이웃)주소
    
        sim_nbrs[i] = new int[left_size]; // 현재 노드의 최대 이웃이   leftsize 만큼이니까 동적생성할때 최대 크기만큼 크기를 잡아서 생성해준다.
        // loop over all neighbors of i
        for (int *j = left_start; j < left_end; j++) {  // 봐라, left_end 까지 계산이 들어가진 않는다. // 포인터가 포인터에 대입이되는거다.
          int nbr_id = *j;
    
          int *right_start = &nbrs[nbr_offs[nbr_id]]; //이웃에대해서 계산
          int *right_end = &nbrs[nbr_offs[nbr_id + 1]];
          int right_size = right_end - right_start;
    
          // compute the similarity
          int num_com_nbrs = get_num_com_nbrs(left_start, left_end, right_start, right_end);
    
          float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));
    
          if (sim > g_epsilon) {
            sim_nbrs[i][num_sim_nbrs[i]] = nbr_id;
            num_sim_nbrs[i]++;
          }
        }
        if (num_sim_nbrs[i] > g_mu) pivots[i] = true;
      }


}

__global__
void step2kernel(bool* d_visited,int* d_cluster_result,int* d_num_sim_nbrs,int* d_pivots,int num_vs){

    
    const int tid = blockDim.x*blockIdx.x + threadIdx.x; // Thread id 
	const int nthread = blockDim.x*gridDim.x;   // Total number of thread
    
    std::fill(d_cluster_result, d_cluster_result + num_vs, -1);
    //int num_clusters = 0;
    for (int i =tid; i<num_vs; i+=nthread) {
      if (!d_pivots[i] || d_visited[i]) continue;
  
      d_visited[i] = true;
      d_cluster_result[i] = i;


      expansion(i, i, d_num_sim_nbrs, sim_nbrs, d_visited, d_pivots, d_cluster_result);
  
      g_num_clusters++;
    }

}
void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {
    
    g_num_vs = num_vs;
    g_num_es = num_es;
    g_epsilon = epsilon;
    g_mu = mu;
    g_nbr_offs = nbr_offs;
    g_nbrs = nbrs;
    __global__ g_num_clusters;
    //int *h_cluster_result = (int*)malloc(sizeof(int)*num_vs);
    
    bool *h_pivots = new bool[num_vs]();  // 동적배열임. 노드가 클러스터의 pivot 인지 아닌지에 대한 불리언 배열
    int *h_num_sim_nbrs = new int[num_vs]();  // 동적배열임. 노드가 클러스터의 pivot이라면 몇개의 neighbor을 가지고있는지에 대한 인트 배열
    sim_nbrs = new int*[num_vs];  //포인터들의 배열이고, 실제 neighbor들을 가리키고있는 배열임.
    
    bool *d_pivots;
    int *d_num_sim_nbrs;
    cudaMalloc(&d_pivots,sizeof(bool)*num_vs);
    cudaMalloc(&d_num_sim_nbrs,sizeof(int)*num_vs);

    cudaMemcpy(d_pivots,h_pivots,sizeof(bool)*num_vs,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_num_sim_nbrs,h_num_sim_nbrs,sizeof(int)*num_vs,cudaMemcpyHostToDevice); 

    step1kernel<<<num_blocks_per_grid,num_threads_per_block>>>>(d_pivots,d_num_sim_nbrs,num_vs,nbrs,nbr_offs);
    
    cudaMemcpy(h_pivots,d_pivots,sizeof(bool)*num_vs,cudaMemcpyDeviceToHost); 
    cudaMemcpy(h_num_sim_nbrs,d_num_sim_nbrs,sizeof(int)*num_vs,cudaMemcpyDeviceToHost); 
    
    cudaDeviceSynchronize();

    
    //step2 start
    bool *h_visited = new bool[num_vs]();

    int* d_cluster_result;
    bool* d_visited;
    cudaMalloc(&d_visited,sizeof(bool)*num_vs);
    cudaMalloc(&d_cluster_result,sizeof(int)*num_vs);

    cudaMemcpy(d_num_sim_nbrs,h_num_sim_nbrs,sizeof(int)*num_vs,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_visited,h_visited,sizeof(bool)*num_vs,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_cluster_result,cluster_result,sizeof(int)*num_vs,cudaMemcpyHostToDevice); 
    cudaMemcpy(d_pivots,h_pivots,sizeof(bool)*num_vs,cudaMemcpyHostToDevice); 

    step2kernel<<<num_blocks_per_grid,num_threads_per_block>>>>(d_visited,d_cluster_result,d_num_sim_nbrs,d_pivots,num_vs);
    cudaMemcpy(h_visited,d_visited,sizeof(bool)*num_vs,cudaMemcpyDeviceToHost); 
    cudaMemcpy(cluster_result,d_cluster_result,sizeof(int)*num_vs,cudaMemcpyDeviceToHost); 
    cudaMemcpy(h_pivots,d_pivots,sizeof(bool)*num_vs,cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
    
    cudaFree(d_visited);
    cudaFree(d_pivots);
    cudaFree(d_cluster_result);
    num_clusters=g_num_clusters;
    
}
