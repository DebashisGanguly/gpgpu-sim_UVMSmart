# UVM Smart

**UVM Smart** is the first repository to provide ***functional*** and ***timing*** simulation support for *Unified Virtual Memory*. This framework extends GPGPU-Sim simulator v3.2 from UBC. Currently, it supports *cudaMallocManaged*, *cudaDeviceSynchronize*, and *cudaMemprefetchAsync*. It includes 10 benchmarks from various benchmark suites (Rodinia, Parboil, Lonestar, Parboil, HPC Challenge). These benchmarks are modified to use UVM APIs.

If you use or build on this framework, please cite:

Debashis Ganguly, Zhiyu Zhang, Jun Yang, Rami Melhem, *Interplay between Hardware Prefetcher and Page Eviction Policy in CPU-GPU Unified Virtual Memory*, The 46th International Symposium on Computer Architecture (ISCA 2019) *[TO APPEAR]*, June 2019.


## Features

 1. A fully-associative TLB and hence TLB look up is performed in a single core cycle, 
 2. A multi-threaded page table walker (configurable page table walk latency),
 3. Workflow for replayable far-fault management (configurable far-fault handling latency),
 4. PCIe transfer latency based on equation derived from curve fitting transfer latency vs transfer size,
 5. PCIe read and write stage queues and transactions (serialized transfers and queueing delay for transaction processing),
 6. Hardware prefetchers (Tree-based neighborhood, Sequential-local 64KB, Random 4KB, On-demand migration),
 7. Page replacement policies (Tree-based neighborhood, Sequential-local 64KB, LRU 4KB, Random 4KB, LRU 2MB, LFU 2MB),
 8. 32-bit access registers per 64KB (basic block),
 9. Delayed migration based on access-counter threshold, 
 10. Rounding up managed allocation and maintaining large-page level full-binary tree. 

Note that currently we do not support heterogeneous systems for CPU-GPU or multi-GPU collaborative workloads. This means CPU page table (validation/invalidation, CPU-mmeory page swapping) is not simulated.

## How to use?

Simple hassle-free. No need to worry about dependencies. Use the Dockerfile in the root directory of the repository. 

```r
sudo docker build -it UVMSmart:latest .
sudo docker run --name <experiment_name> -it UVMSmart:latest
cd /root/gpgpu-sim_UVMSmart/benchmarks/Managed/<benchmark_folder>
vim gpgpusim.config
./run > <output_file>
```

## How to configure?

Currently, we support architectural support for *GeForceGTX 1080Ti* with *PCIe 3.0 16x*. The additional configuration items are added to GeForceGTX1080Ti under configs. Change the respective parameters to simulate desired configuration.

## Copyright Notice

**(c)** *Debashis Ganguly, Department of Computer Science, School of Computing and Information, University of Pittsburgh*, **All rights reserved**.

