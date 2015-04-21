
Lab2.  A Simple CUDA Renderer
-----------------------------

![image](https://github.com/sparkfiresprairie/capl/blob/master/lab2/lab2.png)

In this lab, we write a parallel renderer in CUDA that draws colored circles. While this renderer is very simple, parallelizing the renderer will require us to design and implement data structures that can be efficiently constructed and manipulated in parallel.

###Part1 - CUDA Warm-Up 1: SAXPY (5 pts)
Warm-up task to implement the SAXPY in CUDA. Compare the performance with the sequential CPU-based implementation of SAXPY (time, bandwidth, etc) [[saxpy.cu](./saxpy/saxpy.cu)]

###Part2 - CUDA Warm-Up 2: Parallel Prefix-Sum (10 pts)
In this part, we are asked to come up with parallel implementation of the function find_repeats which, given a list of integers A, returns a list of all indices i for which A[i] == A[i+1]. For example, given the array {1,2,2,1,1,1,3,5,3,3}, our program should output the array {1,3,4,8}. We implement find_repeats by first implementing parallel exclusive prefix-sum operation. 

The following "C-like" code is an iterative version of scan. We use parallel_for to indicate potentially parallel loops.
 
     void exclusive_scan_iterative(int* start, int* end, int* output)
    {
        int N = end - start;
        memmove(output, start, N*sizeof(int));
        // upsweep phase.
        for (int twod = 1; twod < N; twod*=2)
        {
         int twod1 = twod*2;
         parallel_for (int i = 0; i < N; i += twod1)
         {
             output[i+twod1-1] += output[i+twod-1];
         }
        }
    
        output[N-1] = 0;
    
        // downsweep phase.
        for (int twod = N/2; twod >= 1; twod /= 2)
        {
         int twod1 = twod*2;
         parallel_for (int i = 0; i < N; i += twod1)
         {
             int t = output[i+twod-1];
             output[i+twod-1] = output[i+twod1-1];
             output[i+twod1-1] += t; // change twod1 to twod to reverse prefix sum.
         }
        }
    }
    
Code correctness and performance are tested on random input arrays. For reference, a scan score table is provided, showing the performance of a simple CUDA implementation on a stampede cluster with a K20. [[scan.cu](./scan/scan.cu)]

###Part3 - A Simple Circle Renderer (85 pts)
The renderer accepts an array of circles (3D position, velocity, radius, color) as input. The basic sequential algorithm for rendering each frame is:

    Clear image
    for each circle
        update position and velocity
    for each circle
        compute screen bounding box
        for all pixels in bounding box
            compute pixel center point
            if center point is within the circle
                compute color of circle at point
                blend contribution of circle into image for this pixel

The figure below illustrates the basic algorithm for computing circle-pixel coverage using point-in-circle tests. A circle contributes color to an output pixel only if the pixel's center lies within the circle.

![image](https://github.com/sparkfiresprairie/capl/blob/master/lab2/computing_contribution.png)

After familiarizing ourselves with the circle rendering algorithm as implemented in the reference code [[refRenderer.cpp](./render/refRenderer.cpp)], we should deal with CUDA version. The provided CUDA implementation parallelizes computation across all input circles, assigning one circle to each CUDA thread. While this CUDA implementation is a complete implementation of the mathematics of a circle renderer, it contains several major errors that we have to fix. Specifically, the current implementation does not ensure image update is an atomic operation and it does not preserve the required order of image updates.

1. Atomicity: All image update operations must be atomic. The critical region includes reading the four 32-bit floating-point values (the pixel's rgba color), blending the contribution of the current circle with the current image value, and then writing the pixel's color back to memory.

2. Order: Renderer must perform updates to an image pixel in circle input order. That is, if circle 1 and circle 2 both contribute to pixel P, any image updates to P due to circle 1 must be applied to the image before updates to P due to circle 2.

Our job is to write the fastest, correct CUDA renderer implementation we can. 

Our CUDA renderer's [[cudaRenderer.cu](./render/cudaRenderer.cu)] result is as follows:
    
    ------------
    Score table:
    ------------
    -------------------------------------------------------------------------------------------
    | Scene Name      | Naive Time (Tn) | Fast Time (To)  | Your Time (T)   | Score           |
    -------------------------------------------------------------------------------------------
    | rgb             | 4.5             | 1.1240          | 0.7739          | 13              |
    | rand10k         | 230             | 17.1385         | 17.1260         | 13              |
    | rand100k        | 2305            | 171.9914        | 171.7931        | 13              |
    | pattern         | 27              | 0.7801          | 0.7828          | 13              |
    | snowsingle      | 2277            | 46.1337         | 46.4280         | 13              |
    -------------------------------------------------------------------------------------------
                                                          | Total score:    | 65/65           |
    -------------------------------------------------------------------------------------------
