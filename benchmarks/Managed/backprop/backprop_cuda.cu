

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////
extern "C"
void bpnn_copy(float *wf, float *wt, int m, int n);

extern "C"
void bpnn_layerforward(float *l1, float *l2, float *conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float *who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float *w, float *oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
#ifdef GPU  
  float *output_hidden_cuda;
  float *hidden_partial_sum;
  float sum;
  num_blocks = in / 16;  
  dim3  grid( 1 , num_blocks);
  dim3  threads(16 , 16);
  
  cudaMallocManaged((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMallocManaged((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  
#endif

#ifdef CPU

  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);

#endif

#ifdef GPU
 
  printf("Performing GPU computation\n");
  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
  
  
  bpnn_layerforward_CUDA<<< grid, threads >>>(net->input_units,
					      output_hidden_cuda,
					      net->input_weights,
					      hidden_partial_sum,
					      in,
					      hid);
  cudaThreadSynchronize();
  
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
  
     
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {	
      sum += hidden_partial_sum[k * hid + j-1] ;
    }
    sum += net->input_weights[j];
    net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
  #endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

#endif  


#ifdef GPU

  bpnn_adjust_weights_cuda<<< grid, threads >>>(net->hidden_delta,  
						hid, 
						net->input_units, 
						in,
						net->input_weights2, 
						net->input_prev_weights
						);
  cudaDeviceSynchronize();

#define DEBUG
#ifdef DEBUG
  FILE *fp2 = fopen("result.txt","w");
  fprintf(fp2,"Input_units:\n");
  for(int i = 0; i < in + 1; i ++)
    fprintf(fp2,"%f ", net->input_units[i]);
  fprintf(fp2,"\n");
  fprintf(fp2,"Input_weight_one_dim:\n");
  for(int i = 0; i < in + 1; i ++){
    for(int j = 0; j < hid + 1; j++)
      fprintf(fp2,"%f ", net->input_weights2[i*(hid+1)+j]);
    fprintf(fp2,"\n");
  }
  fclose(fp2);
  
#endif

  cudaFree(output_hidden_cuda);
  cudaFree(hidden_partial_sum);
#endif   
  
  
  

}
