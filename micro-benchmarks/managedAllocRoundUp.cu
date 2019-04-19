/* 
** This experiment shows that the user provided size for a managed allocation is rounded up
** to the next 64*(2^i)KB such that full-binary tree with 64KB basic blocks are at the leaf level.
** In this example 2MB192KB is rounded up to 2MB256KB.
*/

#include <iostream>
#include <math.h>
#include <stdio.h>

#define BB (64*1024)
#define LP (2*1024*1024)

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
    for (int i = 0; i < 9; i++) {
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
      } else if (i == 6) {
        x[LP+0*BB] += 'g';
      } else if (i == 7) {
        x[LP+1*BB] += 'h';
      } else if (i == 8) {
        x[LP+2*BB] += 'i';
      }
      sleep(cycles);
    }
  }
} 

int main(void)
{
  int N = LP + 192*1024;
  char *x;
 
  cudaMallocManaged(&x, N*sizeof(char));

  for (int i = 0; i < N; i++) {
    x[i] = 0;
  }
 
  int device = -1;
  cudaGetDevice(&device);

  printf("A 2MB large page allocation is accessed in the following order of 64Kb basic blocks\n");
  printf("x[0*BB]: 0x%llx, x[1*BB]: 0x%llx, x[2*BB]: 0x%llx, x[4*BB]: 0x%llx, x[8*BB]: 0x%llx, x[16*BB]: 0x%llx, x[LP+0*BB]: 0x%llx, x[LP+1*BB]: 0x%llx, x[LP+2*BB]: 0x%llx\n", 
          x+0*BB, x+1*BB, x+2*BB, x+4*BB, x+8*BB, x+16*BB, x+LP+0*BB, x+LP+1*BB, x+LP+2*BB);
  
  getHWPrefetch<<<1, 32>>>(N, x, device);

  cudaDeviceSynchronize();

  printf("After kernel launch the values are:\n");
  printf("x[0*BB] = %c, x[1*BB] = %c, x[2*BB] = %c, x[4*BB] = %c, x[8*BB] = %c, x[16*BB] = %c, x[LP+0*BB] = %c, x[LP+1*BB] = %c, x[LP+2*BB] = %c\n", 
          x[0*BB], x[1*BB], x[2*BB], x[4*BB], x[8*BB], x[16*BB], x[LP+0*BB], x[LP+1*BB], x[LP+2*BB]);

  cudaFree(x);

  return 0;
}

/* Compilation step and profiler output
**
** Look for HtoD PCIe transfers to find the memory address and transfer size to understand the prefetch semantics. 
**
** nvcc -g -Wno-deprecated-gpu-targets -gencode arch=compute_35,code=[compute_35,sm_35] -gencode arch=compute_61,code=[compute_61,sm_61] managedAllocRoundUp.cu -lcudart -o managedAllocRoundUp
**
** nvprof --print-gpu-trace ./managedAllocRoundUp
==5357== NVPROF is profiling process 5357, command: ./managedAllocRoundUp
A 2MB large page allocation is accessed in the following order of 64Kb basic blocks
x[0*BB]: 0x7f1e46000000, x[1*BB]: 0x7f1e46010000, x[2*BB]: 0x7f1e46020000, x[4*BB]: 0x7f1e46040000, x[8*BB]: 0x7f1e46080000, x[16*BB]: 0x7f1e46100000, x[LP+0*BB]: 0x7f1e46200000, x[LP+1*BB]: 0x7f1e46210000, x[LP+2*BB]: 0x7f1e46220000
After kernel launch the values are:
x[0*BB] = a, x[1*BB] = b, x[2*BB] = c, x[4*BB] = d, x[8*BB] = e, x[16*BB] = f, x[LP+0*BB] = g, x[LP+1*BB] = h, x[LP+2*BB] = i
==5357== Profiling application: ./managedAllocRoundUp
==5357== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream        Unified Memory  Virtual Address  Name
477.30ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46000000  [Unified Memory CPU page faults]
477.60ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46010000  [Unified Memory CPU page faults]
477.84ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46020000  [Unified Memory CPU page faults]
478.28ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46040000  [Unified Memory CPU page faults]
479.16ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46080000  [Unified Memory CPU page faults]
480.92ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46100000  [Unified Memory CPU page faults]
484.45ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46200000  [Unified Memory CPU page faults]
484.73ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46210000  [Unified Memory CPU page faults]
484.97ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e46220000  [Unified Memory CPU page faults]
485.32ms  173.73us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46000000  [Unified Memory GPU page faults]
485.33ms  702.84us              (1 1 1)        (32 1 1)         9        0B        0B  GeForce GTX 108         1         7                     -                -  getHWPrefetch(int, char*, int) [108]
485.47ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46000000  [Unified Memory Memcpy HtoD]
485.48ms  5.7600us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46001000  [Unified Memory Memcpy HtoD]
485.49ms  25.504us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46010000  [Unified Memory GPU page faults]
485.51ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46010000  [Unified Memory Memcpy HtoD]
485.51ms  5.8880us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46011000  [Unified Memory Memcpy HtoD]
485.52ms  32.032us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46020000  [Unified Memory GPU page faults]
485.54ms  1.8240us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46020000  [Unified Memory Memcpy HtoD]
485.54ms  11.296us                    -               -         -         -         -  GeForce GTX 108         -         -          124.000000KB   0x7f1e46021000  [Unified Memory Memcpy HtoD]
485.55ms  49.248us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46040000  [Unified Memory GPU page faults]
485.58ms  1.6960us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46040000  [Unified Memory Memcpy HtoD]
485.58ms  21.824us                    -               -         -         -         -  GeForce GTX 108         -         -          252.000000KB   0x7f1e46041000  [Unified Memory Memcpy HtoD]
485.60ms  96.384us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46080000  [Unified Memory GPU page faults]
485.64ms  2.0800us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46080000  [Unified Memory Memcpy HtoD]
485.64ms  54.720us                    -               -         -         -         -  GeForce GTX 108         -         -          508.000000KB   0x7f1e46081000  [Unified Memory Memcpy HtoD]
485.70ms  158.88us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46100000  [Unified Memory GPU page faults]
485.76ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46100000  [Unified Memory Memcpy HtoD]
485.77ms  86.368us                    -               -         -         -         -  GeForce GTX 108         -         -            0.996094MB   0x7f1e46101000  [Unified Memory Memcpy HtoD]
485.86ms  93.568us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46200000  [Unified Memory GPU page faults]
485.94ms  1.6320us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46200000  [Unified Memory Memcpy HtoD]
485.94ms  8.7680us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46201000  [Unified Memory Memcpy HtoD]
485.96ms  25.664us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46210000  [Unified Memory GPU page faults]
485.97ms  1.6960us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46210000  [Unified Memory Memcpy HtoD]
485.97ms  8.8640us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46211000  [Unified Memory Memcpy HtoD]
485.98ms  34.336us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e46220000  [Unified Memory GPU page faults]
486.00ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46220000  [Unified Memory Memcpy HtoD]
486.00ms  14.336us                    -               -         -         -         -  GeForce GTX 108         -         -          124.000000KB   0x7f1e46221000  [Unified Memory Memcpy HtoD]
486.04ms         -                    -               -         -         -         -                -         -         -           PC 0x400cc0   0x7f1e46220000  [Unified Memory CPU page faults]
486.06ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46220000  [Unified Memory Memcpy DtoH]
486.06ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46221000  [Unified Memory Memcpy DtoH]
486.18ms         -                    -               -         -         -         -                -         -         -           PC 0x400cd1   0x7f1e46210000  [Unified Memory CPU page faults]
486.18ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46210000  [Unified Memory Memcpy DtoH]
486.18ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46211000  [Unified Memory Memcpy DtoH]
486.23ms         -                    -               -         -         -         -                -         -         -           PC 0x400ce2   0x7f1e46200000  [Unified Memory CPU page faults]
486.23ms  1.0880us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46200000  [Unified Memory Memcpy DtoH]
486.23ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46201000  [Unified Memory Memcpy DtoH]
486.24ms  5.6000us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7f1e46230000  [Unified Memory Memcpy DtoH]
486.28ms         -                    -               -         -         -         -                -         -         -           PC 0x400cf2   0x7f1e46100000  [Unified Memory CPU page faults]
486.29ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46100000  [Unified Memory Memcpy DtoH]
486.29ms  5.6000us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46101000  [Unified Memory Memcpy DtoH]
486.33ms         -                    -               -         -         -         -                -         -         -           PC 0x400d02   0x7f1e46080000  [Unified Memory CPU page faults]
486.33ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46080000  [Unified Memory Memcpy DtoH]
486.33ms  5.3440us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46081000  [Unified Memory Memcpy DtoH]
486.37ms         -                    -               -         -         -         -                -         -         -           PC 0x400d13   0x7f1e46040000  [Unified Memory CPU page faults]
486.37ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46040000  [Unified Memory Memcpy DtoH]
486.37ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46041000  [Unified Memory Memcpy DtoH]
486.41ms         -                    -               -         -         -         -                -         -         -           PC 0x400d24   0x7f1e46020000  [Unified Memory CPU page faults]
486.41ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46020000  [Unified Memory Memcpy DtoH]
486.41ms  5.3440us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46021000  [Unified Memory Memcpy DtoH]
486.45ms         -                    -               -         -         -         -                -         -         -           PC 0x400d34   0x7f1e46010000  [Unified Memory CPU page faults]
486.46ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46010000  [Unified Memory Memcpy DtoH]
486.46ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46011000  [Unified Memory Memcpy DtoH]
486.50ms         -                    -               -         -         -         -                -         -         -           PC 0x400d3e   0x7f1e46000000  [Unified Memory CPU page faults]
486.50ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e46000000  [Unified Memory Memcpy DtoH]
486.50ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e46001000  [Unified Memory Memcpy DtoH]
486.51ms  5.6960us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7f1e46030000  [Unified Memory Memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
*/
