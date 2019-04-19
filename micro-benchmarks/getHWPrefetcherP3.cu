/* 
** Get the semantics of hardware prefetcher when basic blocks at leaf level are not accessed sequentially
** and fragmented within first 512KB.
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
    for (int i = 0; i < 7; i++) {
      if (i == 0) {
        x[1*BB] += 'a';
      } else if (i == 1) {
        x[3*BB] += 'b';
      } else if (i == 2) {
        x[5*BB] += 'c';
      } else if (i == 3) {
        x[7*BB] += 'd';
      } else if (i == 4) {
        x[0*BB] += 'e';
      } else if (i == 5) {
        x[8*BB] += 'f';
      } else if (i == 6) {
        x[16*BB] += 'g';
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
  printf("x[1*BB]: 0x%llx, x[3*BB]: 0x%llx, x[5*BB]: 0x%llx, x[7*BB]: 0x%llx, x[0*BB]: 0x%llx, x[8*BB]: 0x%llx, x[16*BB]: 0x%llx\n", 
          x+1*BB, x+3*BB, x+5*BB, x+7*BB, x+0*BB, x+8*BB, x+16*BB);
  
  getHWPrefetch<<<1, 32>>>(N, x, device);

  cudaDeviceSynchronize();

  printf("After kernel launch the values are:\n");
  printf("x[1*BB] = %c, x[3*BB] = %c, x[5*BB] = %c, x[7*BB] = %c, x[0*BB] = %c, x[8*BB] = %c, x[16*BB] = %c\n", 
          x[1*BB], x[3*BB], x[5*BB], x[7*BB], x[0*BB], x[8*BB], x[16*BB]);

  cudaFree(x);

  return 0;
}

/* Compilation step and profiler output
**
** Look for HtoD PCIe transfers to find the memory address and transfer size to understand the prefetch semantics. 
**
** nvcc -g -Wno-deprecated-gpu-targets -gencode arch=compute_35,code=[compute_35,sm_35] -gencode arch=compute_61,code=[compute_61,sm_61] getHWPrefetcherP3.cu -lcudart -o getHWPrefetcherP3
**
** nvprof --print-gpu-trace ./getHWPrefetcherP3
==4678== NVPROF is profiling process 4678, command: ./getHWPrefetcherP3
A 2MB large page allocation is accessed in the following order of 64Kb basic blocks
x[1*BB]: 0x7fd472010000, x[3*BB]: 0x7fd472030000, x[5*BB]: 0x7fd472050000, x[7*BB]: 0x7fd472070000, x[0*BB]: 0x7fd472000000, x[8*BB]: 0x7fd472080000, x[16*BB]: 0x7fd472100000
After kernel launch the values are:
x[1*BB] = a, x[3*BB] = b, x[5*BB] = c, x[7*BB] = d, x[0*BB] = e, x[8*BB] = f, x[16*BB] = g
==4678== Profiling application: ./getHWPrefetcherP3
==4678== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream        Unified Memory  Virtual Address  Name
527.84ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fd472000000  [Unified Memory CPU page faults]
528.09ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fd472010000  [Unified Memory CPU page faults]
528.34ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fd472020000  [Unified Memory CPU page faults]
528.78ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fd472040000  [Unified Memory CPU page faults]
529.67ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fd472080000  [Unified Memory CPU page faults]
531.46ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fd472100000  [Unified Memory CPU page faults]
535.07ms  191.94us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fd472010000  [Unified Memory GPU page faults]
535.08ms  575.99us              (1 1 1)        (32 1 1)         9        0B        0B  GeForce GTX 108         1         7                     -                -  getHWPrefetch(int, char*, int) [108]
535.24ms  1.8560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472010000  [Unified Memory Memcpy HtoD]
535.25ms  5.7280us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472011000  [Unified Memory Memcpy HtoD]
535.26ms  24.608us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fd472030000  [Unified Memory GPU page faults]
535.28ms  1.7920us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472030000  [Unified Memory Memcpy HtoD]
535.28ms  5.8560us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472031000  [Unified Memory Memcpy HtoD]
535.29ms  23.616us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fd472050000  [Unified Memory GPU page faults]
535.30ms  1.7920us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472050000  [Unified Memory Memcpy HtoD]
535.30ms  5.9840us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472051000  [Unified Memory Memcpy HtoD]
535.31ms  23.552us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fd472070000  [Unified Memory GPU page faults]
535.33ms  1.7920us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472070000  [Unified Memory Memcpy HtoD]
535.33ms  5.7280us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472071000  [Unified Memory Memcpy HtoD]
535.34ms  58.528us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fd472000000  [Unified Memory GPU page faults]
535.37ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472000000  [Unified Memory Memcpy HtoD]
535.37ms  6.0800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472001000  [Unified Memory Memcpy HtoD]
535.38ms  6.1120us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7fd472020000  [Unified Memory Memcpy HtoD]
535.38ms  6.0800us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7fd472040000  [Unified Memory Memcpy HtoD]
535.39ms  6.0480us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7fd472060000  [Unified Memory Memcpy HtoD]
535.40ms  84.384us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fd472080000  [Unified Memory GPU page faults]
535.43ms  1.7920us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472080000  [Unified Memory Memcpy HtoD]
535.44ms  43.200us                    -               -         -         -         -  GeForce GTX 108         -         -          508.000000KB   0x7fd472081000  [Unified Memory Memcpy HtoD]
535.48ms  158.24us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fd472100000  [Unified Memory GPU page faults]
535.55ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472100000  [Unified Memory Memcpy HtoD]
535.55ms  86.496us                    -               -         -         -         -  GeForce GTX 108         -         -            0.996094MB   0x7fd472101000  [Unified Memory Memcpy HtoD]
535.67ms         -                    -               -         -         -         -                -         -         -           PC 0x400c9f   0x7fd472100000  [Unified Memory CPU page faults]
535.71ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472100000  [Unified Memory Memcpy DtoH]
535.71ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472101000  [Unified Memory Memcpy DtoH]
535.83ms         -                    -               -         -         -         -                -         -         -           PC 0x400caf   0x7fd472080000  [Unified Memory CPU page faults]
535.83ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472080000  [Unified Memory Memcpy DtoH]
535.83ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472081000  [Unified Memory Memcpy DtoH]
535.88ms  1.0240us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472000000  [Unified Memory Memcpy DtoH]
535.88ms         -                    -               -         -         -         -                -         -         -           PC 0x400cb9   0x7fd472000000  [Unified Memory CPU page faults]
535.88ms  5.4080us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472001000  [Unified Memory Memcpy DtoH]
535.92ms  1.0880us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472070000  [Unified Memory Memcpy DtoH]
535.92ms         -                    -               -         -         -         -                -         -         -           PC 0x400cca   0x7fd472070000  [Unified Memory CPU page faults]
535.92ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472071000  [Unified Memory Memcpy DtoH]
535.97ms  1.0240us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472050000  [Unified Memory Memcpy DtoH]
535.97ms         -                    -               -         -         -         -                -         -         -           PC 0x400cdb   0x7fd472050000  [Unified Memory CPU page faults]
535.97ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472051000  [Unified Memory Memcpy DtoH]
536.01ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472030000  [Unified Memory Memcpy DtoH]
536.01ms         -                    -               -         -         -         -                -         -         -           PC 0x400ceb   0x7fd472030000  [Unified Memory CPU page faults]
536.01ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fd472031000  [Unified Memory Memcpy DtoH]
536.05ms         -                    -               -         -         -         -                -         -         -           PC 0x400cfb   0x7fd472010000  [Unified Memory CPU page faults]
536.06ms  1.0880us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fd472010000  [Unified Memory Memcpy DtoH]
536.06ms  10.272us                    -               -         -         -         -  GeForce GTX 108         -         -          124.000000KB   0x7fd472011000  [Unified Memory Memcpy DtoH]
536.08ms  5.6640us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7fd472040000  [Unified Memory Memcpy DtoH]
536.08ms  5.6960us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7fd472060000  [Unified Memory Memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
*/
