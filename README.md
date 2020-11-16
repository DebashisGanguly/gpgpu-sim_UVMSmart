
# UVM Smart

**UVM Smart** is the first repository to provide both ***functional*** and ***timing*** simulation support for *Unified Virtual Memory*. This framework extends GPGPU-Sim v3.2 from UBC. Currently, it supports *cudaMallocManaged*, *cudaDeviceSynchronize*, and *cudaMemprefetchAsync*. It includes 10 benchmarks from various benchmark suites (Rodinia, Parboil, Lonestar, Parboil, HPC Challenge). These benchmarks are modified to use UVM APIs.

If you use or build on this framework, please cite the following papers based on the functionalities you are leveraging.

 1. Please cite the following paper when using prefetches and page
    eviction policies. 
    
    [Debashis Ganguly, Ziyu Zhang, Jun Yang, and Rami Melhem. 2019.
    Interplay between hardware prefetcher and page eviction policy in
    CPU-GPU unified virtual memory. In _Proceedings of the 46th
    International Symposium on Computer Architecture_ (ISCA '19). ACM,
    New York, NY, USA,
    224-235.](https://dl.acm.org/citation.cfm?id=3322224)
 2. Please cite the following paper when using access counter-based
    delayed migration, LFU eviction, cold vs hot data structure classification, and page migration and pinning.
    
    [Debashis Ganguly, Ziyu Zhang, Jun Yang, and Rami Melhem. 2020.
    Adaptive Page Migration for Irregular Data-intensive Applications under GPU Memory Oversubscription. In _Proceedings of the 34th IEEE
    International Parallel &   Distributed Processing Symposium_ (IPDPS
    2020). IEEE , New Orleans, Louisiana, USA, 451-461.](https://ieeexplore.ieee.org/document/9139797)
 3. Please cite the following paper when using adaptive runtime to detect pattern in CPU-GPU interconnect traffic, and policy engine to choose and dynamically employ memory management policies.
    
    Debashis Ganguly, Rami Melhem, and Jun Yang. 2021.
    An Adaptive Framework for Oversubscription Management in CPU-GPU Unified Memory. In _2021 Design, Automation & Test in Europe Conference & Exhibition_ (DATE
    2021).
    
## Features

 1. A fully-associative last-level TLB and hence TLB lookup is performed in a single core cycle, 
 2. A multi-threaded page table (the last level shared) walker (configurable page table walk latency),
 3. Workflow for replayable far-fault management (configurable far-fault handling latency),
 4. PCIe transfer latency based on an equation derived from curve fitting transfer latency vs transfer size,
 5. PCIe read and write stage queues and transactions (serialized transfers and queueing delay for transaction processing),
 6. Prefetchers (Tree-based neighbourhood, Sequential-local 64KB, Random 4KB, On-demand migration),
 7. Page replacement policies (Tree-based neighbourhood, Sequential-local 64KB, LRU 4KB, Random 4KB, LRU 2MB, LFU 2MB),
 8. 32-bit access registers per 64KB (basic block),
 9. Delayed migration based on an access-counter threshold, 
 10. Rounding up managed allocation and maintaining large page (2MB) level full-binary trees.
 11. A runtime engine to detect underlying pattern in CPU-GPU interconnect traffic, a policy engine to choose and dynamically apply the best suitable memory management techniques.

Note that currently, we do not support heterogeneous systems for CPU-GPU or multi-GPU collaborative workloads. This means CPU page table (validation/invalidation, CPU-memory page swapping) is not simulated.

## How to use?

Simple hassle-free. No need to worry about dependencies. Use the Dockerfile in the root directory of the repository. 

```r
sudo docker build -t gpgpu_uvmsmart:latest .
sudo docker run --name <container_name> -it gpgpu_uvmsmart:latest
cd /root/gpgpu-sim_UVMSmart/benchmarks/Managed/<benchmark_folder>
vim gpgpusim.config
./run > <output_file>
sudo docker cp <container_name>:/root/gpgpu-sim_UVMSmart/benchmarks/Managed/<benchmark_folder>/<output_file> .
```

## How to configure?

Currently, we support architectural support for *GeForceGTX 1080Ti* with *PCIe 3.0 16x*. The additional configuration items are added to GeForceGTX1080Ti under configs. Change the respective parameters to simulate the desired configuration.

## What are included?

 1. A set of micro-benchmarks to determine the semantics of prefetcher implemented in *NVIDIA UVM* kernel module (can be found in micro-benchmarks under root).
 2. A micro-benchmark to find out transfer bandwidth for respective transfer size (cudaMemcpy host to device).
 3. A set of benchmarks both with the copy-then-execute model (in Unmanaged under benchmarks folder) and unified virtual memory (in Managed under benchmarks folder).
 4. Specification of the working set, iterations, and the number of kernels launched for managed versions of the benchmarks.
 5. Output log, scripts for plotting, and the derived plots for ISCA'19, IPDPS'20, and DATE'21 papers in Results under benchmarks folder.

## Copyright Notice

Copyright **(c)** 2019
*Debashis Ganguly, Department of Computer Science, School of Computing and Information, University of Pittsburgh*
**All rights reserved**

