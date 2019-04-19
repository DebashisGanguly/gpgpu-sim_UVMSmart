/* 
** Get the semantics of hardware prefetcher for a sequential linear access pattern.
** Only one thread executes with sleeps in between to serialize the transfer 
** such that no two consecutive far-faults are grouped together to decide PCIe transaction.
** Sleep also allows the former PCIe transaction to finish before proceeding with the next transaction.
** This allows the tree to update valid size and make larger prefetch decision.
*/

#include <iostream>
#include <math.h>
#include <stdio.h>

#define BB (64*1024)

__device__
void sleep(clock_t cycles)
{
  clock_t start = clock();
  clock_t now;

  for (;;) {
    now = clock();
    clock_t spent = now > start ? now - start : now + (0xffffffff - start);
    if (spent >= cycles) {
      break;
    }
  }
}

__global__
void getHWPrefetch(int n, char *x, int device)
{
  clock_t cycles = 1000;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index == 0) {
    for (int i = 0; i < 6; i++) {
      if (i == 0) {
        x[0*BB] += 'a';
      } else if (i == 1) {
        x[1*BB] += 'b';
      } else if (i == 2) {
        x[2*BB] += 'c';
      } else if (i == 3) {
        x[4*BB] += 'd';
      } else if (i == 4) {
        x[8*BB] += 'e';
      } else if (i == 5) {
        x[16*BB] += 'f';
      }
      sleep(cycles);
    }
  }
} 

int main(void)
{
  int N = 2*1024*1024;
  char *x;
 
  cudaMallocManaged(&x, N*sizeof(char));

  for (int i = 0; i < N; i++) {
    x[i] = 0;
  }
 
  int device = -1;
  cudaGetDevice(&device);

  printf("A 2MB large page allocation is accessed in the following order of 64Kb basic blocks\n");
  printf("x[0*BB]: 0x%llx, x[1*BB]: 0x%llx, x[2*BB]: 0x%llx, x[4*BB]: 0x%llx, x[8*BB]: 0x%llx, x[16*BB]: 0x%llx\n", 
          x+0*BB, x+1*BB, x+2*BB, x+4*BB, x+8*BB, x+16*BB);
  
  getHWPrefetch<<<1, 32>>>(N, x, device);

  cudaDeviceSynchronize();

  printf("After kernel launch the values are:\n");
  printf("x[0*BB] = %c, x[1*BB] = %c, x[2*BB] = %c, x[4*BB] = %c, x[8*BB] = %c, x[16*BB] = %c\n", 
          x[0*BB], x[1*BB], x[2*BB], x[4*BB], x[8*BB], x[16*BB]);

  cudaFree(x);

  return 0;
}

/* Compilation step and profiler output
**
** Look for HtoD PCIe transfers to find the memory address and transfer size to understand the prefetch semantics. 
**
** nvcc -g -Wno-deprecated-gpu-targets -gencode arch=compute_35,code=[compute_35,sm_35] -gencode arch=compute_61,code=[compute_61,sm_61] getHWPrefetcherP1.cu -lcudart -o getHWPrefetcherP1
**
** nvprof --print-gpu-trace ./getHWPrefetcherP1
==4231== NVPROF is profiling process 4231, command: ./getHWPrefetcherP1
A 2MB large page allocation is accessed in the following order of 64Kb basic blocks
x[0*BB]: 0x7f246e000000, x[1*BB]: 0x7f246e010000, x[2*BB]: 0x7f246e020000, x[4*BB]: 0x7f246e040000, x[8*BB]: 0x7f246e080000, x[16*BB]: 0x7f246e100000
After kernel launch the values are:
x[0*BB] = a, x[1*BB] = b, x[2*BB] = c, x[4*BB] = d, x[8*BB] = e, x[16*BB] = f
==4231== Profiling application: ./getHWPrefetcherP1
==4231== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream        Unified Memory  Virtual Address  Name
681.75ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f246e000000  [Unified Memory CPU page faults]
682.02ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f246e010000  [Unified Memory CPU page faults]
682.26ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f246e020000  [Unified Memory CPU page faults]
682.70ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f246e040000  [Unified Memory CPU page faults]
683.59ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f246e080000  [Unified Memory CPU page faults]
685.33ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f246e100000  [Unified Memory CPU page faults]
688.94ms  183.74us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f246e000000  [Unified Memory GPU page faults]
688.96ms  540.28us              (1 1 1)        (32 1 1)         9        0B        0B  GeForce GTX 108         1         7                     -                -  getHWPrefetch(int, char*, int) [108]
689.11ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e000000  [Unified Memory Memcpy HtoD]
689.11ms  5.7600us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f246e001000  [Unified Memory Memcpy HtoD]
689.13ms  25.216us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f246e010000  [Unified Memory GPU page faults]
689.14ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e010000  [Unified Memory Memcpy HtoD]
689.15ms  5.9840us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f246e011000  [Unified Memory Memcpy HtoD]
689.16ms  31.840us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f246e020000  [Unified Memory GPU page faults]
689.17ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e020000  [Unified Memory Memcpy HtoD]
689.17ms  11.232us                    -               -         -         -         -  GeForce GTX 108         -         -          124.000000KB   0x7f246e021000  [Unified Memory Memcpy HtoD]
689.19ms  49.280us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f246e040000  [Unified Memory GPU page faults]
689.21ms  1.7920us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e040000  [Unified Memory Memcpy HtoD]
689.21ms  21.696us                    -               -         -         -         -  GeForce GTX 108         -         -          252.000000KB   0x7f246e041000  [Unified Memory Memcpy HtoD]
689.24ms  84.384us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f246e080000  [Unified Memory GPU page faults]
689.28ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e080000  [Unified Memory Memcpy HtoD]
689.28ms  43.264us                    -               -         -         -         -  GeForce GTX 108         -         -          508.000000KB   0x7f246e081000  [Unified Memory Memcpy HtoD]
689.32ms  157.02us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f246e100000  [Unified Memory GPU page faults]
689.39ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e100000  [Unified Memory Memcpy HtoD]
689.39ms  86.400us                    -               -         -         -         -  GeForce GTX 108         -         -            0.996094MB   0x7f246e101000  [Unified Memory Memcpy HtoD]
689.51ms         -                    -               -         -         -         -                -         -         -           PC 0x400c90   0x7f246e100000  [Unified Memory CPU page faults]
689.54ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e100000  [Unified Memory Memcpy DtoH]
689.54ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f246e101000  [Unified Memory Memcpy DtoH]
689.66ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e080000  [Unified Memory Memcpy DtoH]
689.66ms         -                    -               -         -         -         -                -         -         -           PC 0x400ca0   0x7f246e080000  [Unified Memory CPU page faults]
689.66ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f246e081000  [Unified Memory Memcpy DtoH]
689.71ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e040000  [Unified Memory Memcpy DtoH]
689.71ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f246e041000  [Unified Memory Memcpy DtoH]
689.71ms         -                    -               -         -         -         -                -         -         -           PC 0x400cb1   0x7f246e040000  [Unified Memory CPU page faults]
689.75ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e020000  [Unified Memory Memcpy DtoH]
689.75ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f246e021000  [Unified Memory Memcpy DtoH]
689.75ms         -                    -               -         -         -         -                -         -         -           PC 0x400cc1   0x7f246e020000  [Unified Memory CPU page faults]
689.79ms  1.0880us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e010000  [Unified Memory Memcpy DtoH]
689.79ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f246e011000  [Unified Memory Memcpy DtoH]
689.80ms         -                    -               -         -         -         -                -         -         -           PC 0x400cd1   0x7f246e010000  [Unified Memory CPU page faults]
689.84ms         -                    -               -         -         -         -                -         -         -           PC 0x400cdb   0x7f246e000000  [Unified Memory CPU page faults]
689.84ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f246e000000  [Unified Memory Memcpy DtoH]
689.84ms  5.2160us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f246e001000  [Unified Memory Memcpy DtoH]
689.85ms  5.6960us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7f246e030000  [Unified Memory Memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
*/
