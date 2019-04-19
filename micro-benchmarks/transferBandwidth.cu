/*
** This program finds out the transfer bandwidth for a given transfer size (cudaMemcpy host to device).
*/

#define PG (4*1024)

int main(void)
{
  int N = 2044*1024;
  float *x, *d_x;

  x = (float*)malloc(N*sizeof(float));
 
  cudaMalloc(&d_x, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 0;
  }

  int current = 0;
  for (int i = 0; i < 9; i++) { 
    cudaMemcpy((d_x+current), (x+current), (int)(1024*pow(2.0,(i+2))), cudaMemcpyHostToDevice);
    current += (int)(1024*pow(2.0,(i+2)));
  }
  
  // Free memory
  cudaFree(d_x);
  free(x);

  return 0;
}

/*
** Visual profiler output
** Look for size and throughput
**
** nvcc -g -Wno-deprecated-gpu-targets -gencode arch=compute_35,code=[compute_35,sm_35] -gencode arch=compute_61,code=[compute_61,sm_61] transferBandwidth.cu -lcudart -o transferBandwidth
**
** nvprof --print-gpu-trace ./transferBandwidth
==6354== NVPROF is profiling process 6354, command: ./transferBandwidth
==6354== Profiling application: ./transferBandwidth
==6354== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
505.78ms  1.1840us                    -               -         -         -         -  4.0000KB  3.2219GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]
505.80ms  1.5680us                    -               -         -         -         -  8.0000KB  4.8657GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]
505.81ms  2.3680us                    -               -         -         -         -  16.000KB  6.4437GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]
505.83ms  3.9690us                    -               -         -         -         -  32.000KB  7.6890GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]
505.87ms  7.2000us                    -               -         -         -         -  64.000KB  8.4771GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]
505.91ms  12.673us                    -               -         -         -         -  128.00KB  9.6323GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]
505.99ms  23.233us                    -               -         -         -         -  256.00KB  10.508GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]
506.13ms  44.418us                    -               -         -         -         -  512.00KB  10.993GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]
506.40ms  87.011us                    -               -         -         -         -  1.0000MB  11.223GB/s    Pageable      Device  GeForce GTX 108         1         7  [CUDA memcpy HtoD]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
SrcMemType: The type of source memory accessed by memory operation/copy
DstMemType: The type of destination memory accessed by memory operation/copy
*/
