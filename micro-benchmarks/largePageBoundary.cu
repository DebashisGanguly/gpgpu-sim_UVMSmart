/*
** This experiment shows that the root of the tree is maximum 2MB or a large page.
** All prefetch decisions are bounded within a 2MB large page.
** For a 6MB allocation, it never crosses 2MB boundary for any prefetch unit.
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
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 6; i++) {
        if (i == 0) {
          x[j*LP+0*BB] += 'a';
        } else if (i == 1) {
          x[j*LP+1*BB] += 'b';
        } else if (i == 2) {
          x[j*LP+2*BB] += 'c';
        } else if (i == 3) {
          x[j*LP+4*BB] += 'd';
        } else if (i == 4) {
          x[j*LP+8*BB] += 'e';
        } else if (i == 5) {
          x[j*LP+16*BB] += 'f';
        }
        sleep(cycles);
      }
    }
  }
} 

int main(void)
{
  int N = 3*LP;
  char *x;
 
  cudaMallocManaged(&x, N*sizeof(char));

  for (int i = 0; i < N; i++) {
    x[i] = 0;
  }
 
  int device = -1;
  cudaGetDevice(&device);

  printf("A 6MB large page allocation is accessed in the following order of 64Kb basic blocks\n");  
  for (int i = 0; i < 3; i++) {
    printf("x[%d*LP+0*BB]: 0x%llx, x[%d*LP+1*BB]: 0x%llx, x[%d*LP+2*BB]: 0x%llx, x[%d*LP+4*BB]: 0x%llx, x[%d*LP+8*BB]: 0x%llx, x[%d*LP+16*BB]: 0x%llx\n", 
            i, x+i*LP+0*BB, i, x+i*LP+1*BB, i, x+i*LP+2*BB, i, x+i*LP+4*BB, i, x+i*LP+8*BB, i, x+i*LP+16*BB);
  }
  
  getHWPrefetch<<<1, 32>>>(N, x, device);

  cudaDeviceSynchronize();

  printf("After kernel launch the values are:\n");
  for (int i = 0; i < 3; i++) {
    printf("x[%d*LP+0*BB] = %c, x[%d*LP+1*BB] = %c, x[%d*LP+2*BB] = %c, x[%d*LP+4*BB] = %c, x[%d*LP+8*BB] = %c, x[%d*LP+16*BB] = %c\n", 
            i, x[i*LP+0*BB], i, x[i*LP+1*BB], i, x[i*LP+2*BB], i, x[i*LP+4*BB], i, x[i*LP+8*BB], i, x[i*LP+16*BB]);
  }

  cudaFree(x);

  return 0;
}

/* Compilation step and profiler output
**
** Look for HtoD PCIe transfers to find the memory address and transfer size to understand the prefetch semantics. 
**
** nvcc -g -Wno-deprecated-gpu-targets -gencode arch=compute_35,code=[compute_35,sm_35] -gencode arch=compute_61,code=[compute_61,sm_61] largePageBoundary.cu -lcudart -o largePageBoundary
**
** nvprof --print-gpu-trace ./largePageBoundary
==4982== NVPROF is profiling process 4982, command: ./largePageBoundary
A 6MB large page allocation is accessed in the following order of 64Kb basic blocks
x[0*LP+0*BB]: 0x7f1e3c000000, x[0*LP+1*BB]: 0x7f1e3c010000, x[0*LP+2*BB]: 0x7f1e3c020000, x[0*LP+4*BB]: 0x7f1e3c040000, x[0*LP+8*BB]: 0x7f1e3c080000, x[0*LP+16*BB]: 0x7f1e3c100000
x[1*LP+0*BB]: 0x7f1e3c200000, x[1*LP+1*BB]: 0x7f1e3c210000, x[1*LP+2*BB]: 0x7f1e3c220000, x[1*LP+4*BB]: 0x7f1e3c240000, x[1*LP+8*BB]: 0x7f1e3c280000, x[1*LP+16*BB]: 0x7f1e3c300000
x[2*LP+0*BB]: 0x7f1e3c400000, x[2*LP+1*BB]: 0x7f1e3c410000, x[2*LP+2*BB]: 0x7f1e3c420000, x[2*LP+4*BB]: 0x7f1e3c440000, x[2*LP+8*BB]: 0x7f1e3c480000, x[2*LP+16*BB]: 0x7f1e3c500000
After kernel launch the values are:
x[0*LP+0*BB] = a, x[0*LP+1*BB] = b, x[0*LP+2*BB] = c, x[0*LP+4*BB] = d, x[0*LP+8*BB] = e, x[0*LP+16*BB] = f
x[1*LP+0*BB] = a, x[1*LP+1*BB] = b, x[1*LP+2*BB] = c, x[1*LP+4*BB] = d, x[1*LP+8*BB] = e, x[1*LP+16*BB] = f
x[2*LP+0*BB] = a, x[2*LP+1*BB] = b, x[2*LP+2*BB] = c, x[2*LP+4*BB] = d, x[2*LP+8*BB] = e, x[2*LP+16*BB] = f
==4982== Profiling application: ./largePageBoundary
==4982== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream        Unified Memory  Virtual Address  Name
507.34ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c000000  [Unified Memory CPU page faults]
507.60ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c010000  [Unified Memory CPU page faults]
507.84ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c020000  [Unified Memory CPU page faults]
508.30ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c040000  [Unified Memory CPU page faults]
509.19ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c080000  [Unified Memory CPU page faults]
510.97ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c100000  [Unified Memory CPU page faults]
514.54ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c200000  [Unified Memory CPU page faults]
514.82ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c210000  [Unified Memory CPU page faults]
515.06ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c220000  [Unified Memory CPU page faults]
515.51ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c240000  [Unified Memory CPU page faults]
516.39ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c280000  [Unified Memory CPU page faults]
518.16ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c300000  [Unified Memory CPU page faults]
521.72ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c400000  [Unified Memory CPU page faults]
522.00ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c410000  [Unified Memory CPU page faults]
522.24ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c420000  [Unified Memory CPU page faults]
522.69ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c440000  [Unified Memory CPU page faults]
523.58ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c480000  [Unified Memory CPU page faults]
525.34ms         -                    -               -         -         -         -                -         -         -           PC 0x400b82   0x7f1e3c500000  [Unified Memory CPU page faults]
529.04ms  188.99us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c000000  [Unified Memory GPU page faults]
529.06ms  1.3566ms              (1 1 1)        (32 1 1)        11        0B        0B  GeForce GTX 108         1         7                     -                -  getHWPrefetch(int, char*, int) [108]
529.21ms  1.9520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c000000  [Unified Memory Memcpy HtoD]
529.22ms  5.8240us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c001000  [Unified Memory Memcpy HtoD]
529.23ms  25.120us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c010000  [Unified Memory GPU page faults]
529.25ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c010000  [Unified Memory Memcpy HtoD]
529.25ms  6.0480us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c011000  [Unified Memory Memcpy HtoD]
529.26ms  32.448us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c020000  [Unified Memory GPU page faults]
529.28ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c020000  [Unified Memory Memcpy HtoD]
529.28ms  11.136us                    -               -         -         -         -  GeForce GTX 108         -         -          124.000000KB   0x7f1e3c021000  [Unified Memory Memcpy HtoD]
529.29ms  48.928us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c040000  [Unified Memory GPU page faults]
529.32ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c040000  [Unified Memory Memcpy HtoD]
529.32ms  21.728us                    -               -         -         -         -  GeForce GTX 108         -         -          252.000000KB   0x7f1e3c041000  [Unified Memory Memcpy HtoD]
529.34ms  83.840us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c080000  [Unified Memory GPU page faults]
529.38ms  1.6960us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c080000  [Unified Memory Memcpy HtoD]
529.38ms  43.392us                    -               -         -         -         -  GeForce GTX 108         -         -          508.000000KB   0x7f1e3c081000  [Unified Memory Memcpy HtoD]
529.43ms  158.18us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c100000  [Unified Memory GPU page faults]
529.49ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c100000  [Unified Memory Memcpy HtoD]
529.49ms  86.272us                    -               -         -         -         -  GeForce GTX 108         -         -            0.996094MB   0x7f1e3c101000  [Unified Memory Memcpy HtoD]
529.59ms  57.664us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c200000  [Unified Memory GPU page faults]
529.63ms  1.9520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c200000  [Unified Memory Memcpy HtoD]
529.64ms  5.7920us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c201000  [Unified Memory Memcpy HtoD]
529.65ms  22.080us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c210000  [Unified Memory GPU page faults]
529.66ms  1.8240us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c210000  [Unified Memory Memcpy HtoD]
529.66ms  5.7280us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c211000  [Unified Memory Memcpy HtoD]
529.67ms  31.424us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c220000  [Unified Memory GPU page faults]
529.69ms  1.9520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c220000  [Unified Memory Memcpy HtoD]
529.69ms  11.200us                    -               -         -         -         -  GeForce GTX 108         -         -          124.000000KB   0x7f1e3c221000  [Unified Memory Memcpy HtoD]
529.70ms  54.848us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c240000  [Unified Memory GPU page faults]
529.73ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c240000  [Unified Memory Memcpy HtoD]
529.73ms  21.952us                    -               -         -         -         -  GeForce GTX 108         -         -          252.000000KB   0x7f1e3c241000  [Unified Memory Memcpy HtoD]
529.76ms  83.456us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c280000  [Unified Memory GPU page faults]
529.79ms  1.7920us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c280000  [Unified Memory Memcpy HtoD]
529.80ms  43.232us                    -               -         -         -         -  GeForce GTX 108         -         -          508.000000KB   0x7f1e3c281000  [Unified Memory Memcpy HtoD]
529.84ms  155.10us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c300000  [Unified Memory GPU page faults]
529.90ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c300000  [Unified Memory Memcpy HtoD]
529.91ms  86.336us                    -               -         -         -         -  GeForce GTX 108         -         -            0.996094MB   0x7f1e3c301000  [Unified Memory Memcpy HtoD]
530.00ms  55.360us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c400000  [Unified Memory GPU page faults]
530.04ms  1.7600us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c400000  [Unified Memory Memcpy HtoD]
530.05ms  5.8880us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c401000  [Unified Memory Memcpy HtoD]
530.06ms  21.728us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c410000  [Unified Memory GPU page faults]
530.07ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c410000  [Unified Memory Memcpy HtoD]
530.07ms  5.7600us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c411000  [Unified Memory Memcpy HtoD]
530.08ms  30.304us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c420000  [Unified Memory GPU page faults]
530.09ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c420000  [Unified Memory Memcpy HtoD]
530.10ms  11.072us                    -               -         -         -         -  GeForce GTX 108         -         -          124.000000KB   0x7f1e3c421000  [Unified Memory Memcpy HtoD]
530.11ms  48.032us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c440000  [Unified Memory GPU page faults]
530.13ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c440000  [Unified Memory Memcpy HtoD]
530.13ms  21.760us                    -               -         -         -         -  GeForce GTX 108         -         -          252.000000KB   0x7f1e3c441000  [Unified Memory Memcpy HtoD]
530.16ms  82.368us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c480000  [Unified Memory GPU page faults]
530.19ms  1.9520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c480000  [Unified Memory Memcpy HtoD]
530.20ms  43.232us                    -               -         -         -         -  GeForce GTX 108         -         -          508.000000KB   0x7f1e3c481000  [Unified Memory Memcpy HtoD]
530.24ms  154.43us                    -               -         -         -         -  GeForce GTX 108         -         -                     1   0x7f1e3c500000  [Unified Memory GPU page faults]
530.30ms  1.7280us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c500000  [Unified Memory Memcpy HtoD]
530.30ms  86.432us                    -               -         -         -         -  GeForce GTX 108         -         -            0.996094MB   0x7f1e3c501000  [Unified Memory Memcpy HtoD]
530.42ms         -                    -               -         -         -         -                -         -         -           PC 0x400d41   0x7f1e3c100000  [Unified Memory CPU page faults]
530.44ms  1.2160us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c100000  [Unified Memory Memcpy DtoH]
530.44ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c101000  [Unified Memory Memcpy DtoH]
530.57ms         -                    -               -         -         -         -                -         -         -           PC 0x400d5f   0x7f1e3c080000  [Unified Memory CPU page faults]
530.57ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c080000  [Unified Memory Memcpy DtoH]
530.57ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c081000  [Unified Memory Memcpy DtoH]
530.61ms         -                    -               -         -         -         -                -         -         -           PC 0x400d7d   0x7f1e3c040000  [Unified Memory CPU page faults]
530.61ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c040000  [Unified Memory Memcpy DtoH]
530.62ms  5.3440us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c041000  [Unified Memory Memcpy DtoH]
530.66ms         -                    -               -         -         -         -                -         -         -           PC 0x400d9a   0x7f1e3c020000  [Unified Memory CPU page faults]
530.66ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c020000  [Unified Memory Memcpy DtoH]
530.66ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c021000  [Unified Memory Memcpy DtoH]
530.71ms         -                    -               -         -         -         -                -         -         -           PC 0x400db7   0x7f1e3c010000  [Unified Memory CPU page faults]
530.72ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c010000  [Unified Memory Memcpy DtoH]
530.72ms  5.3440us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c011000  [Unified Memory Memcpy DtoH]
530.76ms         -                    -               -         -         -         -                -         -         -           PC 0x400dcd   0x7f1e3c000000  [Unified Memory CPU page faults]
530.76ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c000000  [Unified Memory Memcpy DtoH]
530.76ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c001000  [Unified Memory Memcpy DtoH]
530.77ms  5.6640us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7f1e3c030000  [Unified Memory Memcpy DtoH]
530.82ms         -                    -               -         -         -         -                -         -         -           PC 0x400d41   0x7f1e3c300000  [Unified Memory CPU page faults]
530.83ms  1.2160us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c300000  [Unified Memory Memcpy DtoH]
530.83ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c301000  [Unified Memory Memcpy DtoH]
530.87ms         -                    -               -         -         -         -                -         -         -           PC 0x400d5f   0x7f1e3c280000  [Unified Memory CPU page faults]
530.87ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c280000  [Unified Memory Memcpy DtoH]
530.87ms  5.3440us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c281000  [Unified Memory Memcpy DtoH]
530.91ms         -                    -               -         -         -         -                -         -         -           PC 0x400d7d   0x7f1e3c240000  [Unified Memory CPU page faults]
530.91ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c240000  [Unified Memory Memcpy DtoH]
530.91ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c241000  [Unified Memory Memcpy DtoH]
530.95ms         -                    -               -         -         -         -                -         -         -           PC 0x400d9a   0x7f1e3c220000  [Unified Memory CPU page faults]
530.95ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c220000  [Unified Memory Memcpy DtoH]
530.96ms  5.2160us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c221000  [Unified Memory Memcpy DtoH]
531.00ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c210000  [Unified Memory Memcpy DtoH]
531.00ms         -                    -               -         -         -         -                -         -         -           PC 0x400db7   0x7f1e3c210000  [Unified Memory CPU page faults]
531.00ms  5.1840us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c211000  [Unified Memory Memcpy DtoH]
531.04ms         -                    -               -         -         -         -                -         -         -           PC 0x400dcd   0x7f1e3c200000  [Unified Memory CPU page faults]
531.05ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c200000  [Unified Memory Memcpy DtoH]
531.05ms  5.2160us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c201000  [Unified Memory Memcpy DtoH]
531.05ms  5.6320us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7f1e3c230000  [Unified Memory Memcpy DtoH]
531.10ms         -                    -               -         -         -         -                -         -         -           PC 0x400d41   0x7f1e3c500000  [Unified Memory CPU page faults]
531.11ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c500000  [Unified Memory Memcpy DtoH]
531.11ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c501000  [Unified Memory Memcpy DtoH]
531.15ms         -                    -               -         -         -         -                -         -         -           PC 0x400d5f   0x7f1e3c480000  [Unified Memory CPU page faults]
531.15ms  1.0560us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c480000  [Unified Memory Memcpy DtoH]
531.15ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c481000  [Unified Memory Memcpy DtoH]
531.19ms  1.0880us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c440000  [Unified Memory Memcpy DtoH]
531.19ms         -                    -               -         -         -         -                -         -         -           PC 0x400d7d   0x7f1e3c440000  [Unified Memory CPU page faults]
531.19ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c441000  [Unified Memory Memcpy DtoH]
531.24ms  1.1840us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c420000  [Unified Memory Memcpy DtoH]
531.24ms         -                    -               -         -         -         -                -         -         -           PC 0x400d9a   0x7f1e3c420000  [Unified Memory CPU page faults]
531.24ms  5.2800us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c421000  [Unified Memory Memcpy DtoH]
531.28ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c410000  [Unified Memory Memcpy DtoH]
531.28ms         -                    -               -         -         -         -                -         -         -           PC 0x400db7   0x7f1e3c410000  [Unified Memory CPU page faults]
531.28ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c411000  [Unified Memory Memcpy DtoH]
531.32ms         -                    -               -         -         -         -                -         -         -           PC 0x400dcd   0x7f1e3c400000  [Unified Memory CPU page faults]
531.33ms  1.1520us                    -               -         -         -         -  GeForce GTX 108         -         -            4.000000KB   0x7f1e3c400000  [Unified Memory Memcpy DtoH]
531.33ms  5.3120us                    -               -         -         -         -  GeForce GTX 108         -         -           60.000000KB   0x7f1e3c401000  [Unified Memory Memcpy DtoH]
531.33ms  5.6960us                    -               -         -         -         -  GeForce GTX 108         -         -           64.000000KB   0x7f1e3c430000  [Unified Memory Memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
*/
