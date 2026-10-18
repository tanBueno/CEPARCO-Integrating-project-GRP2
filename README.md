# CEPARCO-Integrating-project-GRP2

# Video pitch link
https://drive.google.com/file/d/1Lac1JPG2trMI-AG0c_r0rROZRKeCm6Y-/view?usp=sharing 

## 0.) Abstract of your proposed project

This project implements a GPU-accelerated edge detection algorithm using CUDA to implement parallel processing and compares its performance with a sequential C implementation. The process is done by applying a 5x5 Gaussian filter to reduce image noise, followed by a 3x3 Sobel operator to detect edges by calculating pixel gradients. Since each pixel is processed independently during convolution, the workload is distributed across GPU threads using 16x16 thread blocks. Optimizations such as shared memory tiling and constant memory are explored to improve memory access, while hysteresis thresholding remains on the CPU due to its dependence on pixel connectivity. The implementation will be tested on grayscale images of different resolutions to compare the execution times of the sequential C baseline and CUDA pipelines.

## 1.) Description of your proposed project

## a. What are the inputs to your project? 

Input Image Data: Grayscale images (.bmp/ .png/ .pgm) across different resolutions to evaluate scalability.
Filter Kernels and Parameters:
5x5 2D Gaussian low-pass filter kernel parameterized by standard deviation
3x3 Horizontal and Vertical Sobel derivative kernels 
Dual Hysteresis for sequential host side edge detection

## b. Propose process

## i. What processes will be parallelized using SIMT (GPU/CUDA) or SIMD-based AVX/AVX2/AVX512 assembly project?

The parallel pipeline consists of two back to back 2D spatial convolution kernels executed in GPU VRAM:
Stage 1: 2D 5x5 Gaussian Low Pass Convolution Kernel: Each GPU thread suppress high frequency spatial noise by convolving a 5x5 local neighborhood around pixel with the Gaussian kernel:

<img width="487" height="67" alt="image" src="https://github.com/user-attachments/assets/a11e71f8-fa96-44b4-bb71-26c524d6f79a" />


Stage 2: 3x3 Sobel Gradient Magnitude & Direction Kerne: Reading directly from the intermediate device buffer Ismooth, each GPU thread computes the horizontal (Gx) and vertical (Gy) first order spatial derivative and evaluates the total gradient magnitude M(x,y):

<img width="545" height="306" alt="image" src="https://github.com/user-attachments/assets/4013d3a3-1632-4fac-b380-d4997ee7d3ea" />


## ii. What existing implementations have been done with the proposed process (i.e., related works)? How would it differ from your proposed implementation if it has been parallelized? 

Existing Implementations (Related work)
Sequential CPU Approach: Standard C implementations use nested for loops to process an image one pixel at a time. For high resolution images, calculating a 5x5 Gaussian blur followed by a 3x3 Sobel filter requires millions of iterations which make the CPU execution time very slow.

High Level libraries and basic GPU Approaches: Libraries like OpenCV provide prebuilt functions, but they hide the underlying parallel architecture. Meanwhile CUDA implementations assign one GPU thread per pixel and real all neighboring pixels directly from slow GPU VRAM. Because neighboring pixels have overlapping windows, each pixel gets fetched from slow memory up to 25 times during the 5x5 Gaussian stage and 9 times during the 3x3 Sobel stage, creating a heavy memory bottleneck.

Proposed Differentiation: 
Instead of relying on prebuilt libraries or slow VRAM reads, we will build a custom Shared Memory Tiled CUDA kernel. Threads in each block will work together to load an image tile into fast on chip Shared memory once, and store the fixed filter weights in Constant Memory so all threads can read them instantly
Three Way Performance Benchmark: We will directly compare the execution times of three versions. First is Sequential C baseline, Second is a Naive CUDA kernel using Global Memory, and Last is Our Optimized CUDA kernel using shared Memory.
Single Channel VRAM Pipeline: By focusing strictly on 8 bit grayscale images, we reduce CPU to GPU data transfer size by 3x compared to RGB. We also keep the blurred image directly inside GPU memory between Stage 1 and 2, avoiding extra data transfers back and fourth to the CPU



## iii. Discuss plans on how to parallelize your proposed process.

To parallelize the edge detection pipeline in CUDA, we plan to divide the input image into a 2D grid of 16x16 thread blocks so that the GPU Thread is responsible for computing a single pixel. To apply the Gaussian blur and Sobel Filters, it is necessary to make nested loops that would scan the pixels one by one, which would be slower the larger the image gets. We use the GPU to compute thousands of pixels and their convolutions at the same time. We also plan to explore using CUDA shared memory and constant memory so threads in the same block can reuse neighboring pixel values and filter weights without having to repeatedly fetch them from slower global memory. Keeping the image data inside the GPU memory between grayscale, Gaussian blur, and Sobel stages is done so that we only have to transfer the image between the CPU and GPU at the start and end of the pipeline.

## iv. What other processes are in the project but will not be parallelized.

The supporting parts of the program will remain on the CPU since they do not benefit from GPU parallelization. This includes reading the input image file from disk, decoding the image format into a pixel array, and saving the final edge-detected image back to a file. Setting up the memory and copying the image data between the host CPU and the GPU, will be handled by the main C program.

## c. What are the outputs of your project?

The project outputs will be the visual outputs of the Gaussian image, Sobel gradient magnitude map, and threshold of the binary edge map. The CUDA outputs will be matched with the sequential C baseline using mean squared error and a IEEE-754 floating-point tolerance. A performance test benchmark report will be added. This will show the execution time comparisons and speedup tables for images from 5122 up to 8K resolution. These comparisons will be with the following, Sequential C, Naive CUDA kernel, and the Optimized CUDA kernel that has the shared memory tiling and constant memory.

## 2.) Proposed target implementation platform of the proposed project

The target implementation platform for the proposed project are the following two: GPU Hardware to be used is an NVIDIA GPU with CUDA Compute Capability 7.5+. A plausible device to be used is a laptop with an NVIDIA RTX series. The other platform is the CPU x86-64 multi-core processor that will host the main code. This will execute the kernel call to CUDA accessing the GPU threads.

## 3.) Mid-project milestones of the proposed project

The following objectives are expected by the group to be completed: The first objective is a complete C program that handles the input and output of the image. It will run a baseline sequential 2D Gaussian smoothing and 2D edge detection as a comparison for the parallelized implementation, and an expected output of results of CPU execution times. The second objective is the CUDA code. The implementation of the Gaussian Blur and the Sobel kernels that will use the 2D thread blocks and global memory. The third objective is a detailed comparison of the base line code to the CUDA implementation. This consists of comparing the runtimes of the baseline C implementation to the GPU kernel runtime. A stress test will be performed with a 1080p and 4k image test.

## 4. References

[1] R. C. Gonzalez and R. E. Woods, Digital Image Processing, 4th ed. New York, NY, USA: Pearson, 2018. (Covers the mathematical formulation of 2D Gaussian low-pass spatial filtering and Sobel gradient operators).

[2] D. B. Kirk and W. W. Hwu, Programming Massively Parallel Processors: A Hands-on Approach, 4th ed. Cambridge, MA, USA: Morgan Kaufmann, 2022, ch. 7, pp. 139–168. (Covers 2D parallel convolution in CUDA, Constant Memory caching, and Shared Memory tiling with halo cells). 

[3] J. Canny, "A computational approach to edge detection," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-8, no. 6, pp. 679–698, Nov. 1986, doi: 10.1109/TPAMI.1986.4767851. (Foundational paper for multi-stage Gaussian smoothing, gradient magnitude extraction, and hysteresis edge tracking).

[4] Y. Luo and R. Duraiswami, "Canny edge detection on NVIDIA CUDA," in Proc. IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Anchorage, AK, USA, Jun. 2008, pp. 1–8, doi: 10.1109/CVPRW.2008.4563088. (Related work demonstrating GPU-accelerated Gaussian and Sobel stages vs. sequential host bottlenecks in hysteresis tracking).

[5] K. Che, W. Wang, and L. Zhang, "A parallel implementation of Sobel edge detection based on CUDA," in Proc. International Conference on Computer Science and Service System (CSSS), Nanjing, China, Aug. 2012, pp. 1882–1885. (Related work analyzing global memory bandwidth bottlenecks in naive CUDA Sobel kernels).

[6] J. L. Hennessy and D. A. Patterson, Computer Architecture: A Quantitative Approach, 6th ed. Cambridge, MA, USA: Morgan Kaufmann, 2017, ch. 4. (Course textbook reference for Data-Level Parallelism in Vector, SIMD, and GPU/SIMT architectures).

[7] NVIDIA Corporation, CUDA C++ Programming Guide, Version 12.x, 2025. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

