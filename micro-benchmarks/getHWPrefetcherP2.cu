/* 
** Get the semantics of hardware prefetcher when basic blocks at leaf level are not accessed sequentially.
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
        x[1*BB] += 'a';
      } else if (i == 1) {
        x[3*BB] += 'b';
      } else if (i == 2) {
        x[0*BB] += 'c';
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
  printf("x[1*BB]: 0x%llx, x[3*BB]: 0x%llx, x[0*BB]: 0x%llx, x[4*BB]: 0x%llx, x[8*BB]: 0x%llx, x[16*BB]: 0x%llx\n", 
          x+1*BB, x+3*BB, x+0*BB, x+4*BB, x+8*BB, x+16*BB);
  
  getHWPrefetch<<<1, 32>>>(N, x, device);

  cudaDeviceSynchronize();

  printf("After kernel launch the values are:\n");
  printf("x[1*BB] = %c, x[3*BB] = %c, x[0*BB] = %c, x[4*BB] = %c, x[8*BB] = %c, x[16*BB] = %c\n", 
          x[1*BB], x[3*BB], x[0*BB], x[4*BB], x[8*BB], x[16*BB]);

  cudaFree(x);

  return 0;
}

/* Compilation step and profiler output
**
** Look for HtoD PCIe transfers to find the memory address and transfer size to understand the prefetch semantics. 
**
** nvcc -g -Wno-deprecated-gpu-targets -gencode arch=compute_35,code=[compute_35,sm_35] -gencode arch=compute_61,code=[compute_61,sm_61] getHWPrefetcherP2.cu -lcudart -o getHWPrefetcherP2
**
** nvprof --print-gpu-trace ./getHWPrefetcherP2
==4347== NVPROF is profiling process 4347, command: ./getHWPrefetcherP2
A 2MB large page allocation is accessed in the following order of 64Kb basic blocks
x[1*BB]: 0x7fa016010000, x[3*BB]: 0x7fa016030000, x[0*BB]: 0x7fa016000000, x[4*BB]: 0x7fa016040000, x[8*BB]: 0x7fa016080000, x[16*BB]: 0x7fa016100000
After kernel launch the values are:
x[1*BB] = a, x[3*BB] = b, x[0*BB] = c, x[4*BB] = d, x[8*BB] = e, x[16*BB] = f
==4347== Profiling application: ./getHWPrefetcherP2
==4347== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream        Unified Memory  Virtual Address  Name
496.26ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fa016000000  [Unified Memory CPU page faults]
496.51ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fa016010000  [Unified Memory CPU page faults]
496.75ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fa016020000  [Unified Memory CPU page faults]
497.20ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fa016040000  [Unified Memory CPU page faults]
498.08ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fa016080000  [Unified Memory CPU page faults]
499.82ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7fa016100000  [Unified Memory CPU page faults]
503.45ms  175.84us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fa016010000  [Unified Memory GPU page faults]
503.46ms  549.46us              (1 1 1)        (32 1 1)         9        0B        0B  GeForce GTX 108         1         7                     -                -  getHWPrefetch(int, char*, int) [108]
503.61ms  1.5680us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016010000  [Unified Memory Memcpy HtoD]
503.61ms  5.7280us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fa016011000  [Unified Memory Memcpy HtoD]
503.62ms  25.568us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fa016030000  [Unified Memory GPU page faults]
503.64ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016030000  [Unified Memory Memcpy HtoD]
503.64ms  5.7280us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fa016031000  [Unified Memory Memcpy HtoD]
503.65ms  47.840us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fa016000000  [Unified Memory GPU page faults]
503.68ms  1.6960us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016000000  [Unified Memory Memcpy HtoD]
503.68ms  5.7920us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fa016001000  [Unified Memory Memcpy HtoD]
503.68ms  12.224us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7fa016020000  [Unified Memory Memcpy HtoD]
503.70ms  50.272us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fa016040000  [Unified Memory GPU page faults]
503.72ms  1.8240us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016040000  [Unified Memory Memcpy HtoD]
503.72ms  21.856us                    -               -         -         -         -  GeForce GTX 108         -         -          252.000000KB   0x7fa016041000  [Unified Memory Memcpy HtoD]
503.75ms  84.800us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fa016080000  [Unified Memory GPU page faults]
503.79ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016080000  [Unified Memory Memcpy HtoD]
503.79ms  43.584us                    -               -         -         -         -  GeForce GTX 108         -         -          508.000000KB   0x7fa016081000  [Unified Memory Memcpy HtoD]
503.84ms  156.42us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7fa016100000  [Unified Memory GPU page faults]
503.90ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016100000  [Unified Memory Memcpy HtoD]
503.90ms  86.432us                    -               -         -         -         -  GeForce GTX 108         -         -            0.996094MB   0x7fa016101000  [Unified Memory Memcpy HtoD]
504.02ms         -                    -               -         -         -         -                -         -         -           PC 0x400c90   0x7fa016100000  [Unified Memory CPU page faults]
504.07ms  1.0240us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016100000  [Unified Memory Memcpy DtoH]
504.07ms  5.5360us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fa016101000  [Unified Memory Memcpy DtoH]
504.18ms         -                    -               -         -         -         -                -         -         -           PC 0x400ca0   0x7fa016080000  [Unified Memory CPU page faults]
504.19ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016080000  [Unified Memory Memcpy DtoH]
504.19ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fa016081000  [Unified Memory Memcpy DtoH]
504.23ms         -                    -               -         -         -         -                -         -         -           PC 0x400cb1   0x7fa016040000  [Unified Memory CPU page faults]
504.23ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016040000  [Unified Memory Memcpy DtoH]
504.23ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fa016041000  [Unified Memory Memcpy DtoH]
504.27ms         -                    -               -         -         -         -                -         -         -           PC 0x400cbb   0x7fa016000000  [Unified Memory CPU page faults]
504.27ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016000000  [Unified Memory Memcpy DtoH]
504.27ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fa016001000  [Unified Memory Memcpy DtoH]
504.31ms         -                    -               -         -         -         -                -         -         -           PC 0x400ccb   0x7fa016030000  [Unified Memory CPU page faults]
504.32ms  1.4080us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016030000  [Unified Memory Memcpy DtoH]
504.32ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7fa016031000  [Unified Memory Memcpy DtoH]
504.36ms         -                    -               -         -         -         -                -         -         -           PC 0x400cdb   0x7fa016010000  [Unified Memory CPU page faults]
504.36ms  1.0880us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7fa016010000  [Unified Memory Memcpy DtoH]
504.36ms  10.560us                    -               -         -         -         -  GeForce GTX 108         -         -          124.000000KB   0x7fa016011000  [Unified Memory Memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
*/
