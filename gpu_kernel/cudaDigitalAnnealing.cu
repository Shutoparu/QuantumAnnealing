#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

texture<int, 1, cudaReadModeElementType> b_text;
texture<float, 1, cudaReadModeElementType> Q_text;

/**
 * @brief used to check if cuda code goes wrong
 */
void checkCudaError () {
    cudaError_t err = cudaGetLastError ();
    if (err != cudaSuccess) {

        printf ("Cuda Error: %s, %s\n", cudaGetErrorName (err), cudaGetErrorString (err));
        exit (1);
    }
}

/**
 * @brief sum up the given aray
 *
 * @param arr input array
 * @param size the size of the array
 * @return the sum of the array
 */
float sum (float* arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

/**
 * @param arr input binary array
 * @param size size of the array
 * @return the index of a random non-zero value from the array
 */
int randChoose (float* arr, int size) {

    int nonZeroNum = 0;

    int* indicies;
    indicies = (int*)malloc (size * sizeof (int));

    for (int i = 0; i < size; i++) {
        if (arr[i] != 0) {
            indicies[nonZeroNum] = i;
            nonZeroNum++;
        }
    }

    int index = indicies[rand () % nonZeroNum];
    free (indicies);

    return index;
}

/**
 * @brief find the minimum value of the given array
 *
 * @param arr input array
 * @param size the size of the array
 * @return return the minimum value of the array
 */
float max (float* arr, int size) {
    float max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

/**
 * @brief calculate the energy with given qubo matrix and binary state
 *
 * @param b array representing binary
 * @param Q qubo matrix
 * @param tempArr a temporary array to store the dot product of b^T * (Q*b)
 * @param dim dimention of the array and matrix
 */
__global__ void calculateEnergy (int* b, float* Q, float* tempArr, int dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim) {
        tempArr[i] = 0;
        for (int n = 0; n < dim; n++) {
            tempArr[i] += Q[i * dim + n] * b[n];
        }
        tempArr[i] = tempArr[i] * b[i];
    }
}

__global__ void dot1 (float* out, int dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim) {
        out[i] = 0;
        for (int j = 0; j < dim; j++) {
            out[i] += tex1Dfetch (b_text, j) * tex1Dfetch (Q_text, dim * i + j);
        }
    }
}

__global__ void dot2 (float* out, int dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim) {
        out[i] *= tex1Dfetch (b_text, i);
    }
}

/**
 * @brief calculate the energy change per bit flip, record the result and return an array of the result
 *
 * @param b_copy the binary array
 * @param Q the qubo matrix
 * @param dim the dimention of the matrix and array
 * @param offset constant to deduct if the result was not accepted in the previous round
 * @param beta a factor to accept randomness
 * @param stat the array to be returned, include [0] acceptance and [1] energy change
 * @param seed a seed to create random float between (0,1] in kernel
 */
__global__ void slipBinary (int dim, float offset, float beta, float* stat, float seed) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim - 1) {
        int flipped = 0;
        float delta_E;
        curandState state;
        curand_init (seed, i, 0, &state);

        // get energy change for flipping the bit [i] (check delta_E)

        // check flip
        if (tex1Dfetch (b_text, i) == 0) {
            flipped = 1;
        }

        for (int n = 0; n < dim; n++) {
            if (n == i && flipped == 1) {
                delta_E += tex1Dfetch (Q_text, i * dim + n); // time consuming
            } else {
                delta_E += tex1Dfetch (b_text, n) * tex1Dfetch (Q_text, i * dim + n); // time consuming
            }
        }

        if (flipped != 0) {
            delta_E = 2 * delta_E - tex1Dfetch (Q_text, i * dim + i) - offset;
        } else {
            delta_E = -2 * delta_E + tex1Dfetch (Q_text, i * dim + i) - offset;
        }

        // check energy or check % (check pass)
        float p = exp (-delta_E * beta);
        if (abs (delta_E) <= 0.0000005) {
            // stat[i] = curand_uniform (&state) > 0.4 ? 1 : 0;
            stat[i] = 0;
        } else if (p > curand_uniform (&state)) {
            stat[i] = 1;
            // } else if (abs (delta_E) <= 0.0000005) {
            //     // stat[i] = curand_uniform (&state) > 0.4 ? 1 : 0;
            //     stat[i] = 0;
            // } else if (delta_E < 0) {
            //     stat[i] = 1;
        } else {
            stat[i] = 0;
        }
        stat[dim + i] = delta_E;
    }
}

/**
 * @brief create the beta array
 *
 * @param betaStart starting value of beta
 * @param betaStop ending value of beta
 * @param beta the beta array to be returned
 * @param sweeps the length of beta array
 */
void getAnnealingBeta (float betaStart, float betaStop, float* beta, int sweeps) {

    float logBetaStart = log (betaStart);
    float logBetaStop = log (betaStop);
    float logBetaRange = (logBetaStop - logBetaStart) / (float)sweeps;
    for (int i = 0; i < sweeps; i++) {
        beta[i] = exp (logBetaStart + logBetaRange * i);
    }
}

// /**
//  * @brief the function that runs the digital annealing algorithm
//  *
//  * @param b binary array
//  * @param Q qubo matrix
//  * @param dim dimention of binary array and qubo matrix
//  * @param energy energy matrix to be returned, will record energy after per flip
//  * @param sweeps number of iterations to be done
//  */
// void digitalAnnealing (int* b, float* Q, int dim, float* energy, int sweeps) {

//     int blocks = 32 * 8;
//     int threads = dim / blocks + 1;

//     int betaStart = 1;
//     int betaStop = 50;

//     float* beta;
//     beta = (float*)malloc (sweeps * sizeof (float));
//     getAnnealingBeta (betaStart, betaStop, beta, sweeps);

//     float offset = 0;
//     float offsetIncreasingRate = 0.1;

//     float* stat;
//     cudaMalloc (&stat, 2 * dim * sizeof (float));

//     float* stat_host;
//     cudaMallocHost (&stat_host, 2 * dim * sizeof (float));

//     int* b_copy;
//     cudaMalloc (&b_copy, dim * sizeof (int));

//     float* Q_copy;
//     cudaMalloc (&Q_copy, dim * dim * sizeof (float));
//     cudaMemcpy (Q_copy, Q, dim * dim * sizeof (float), cudaMemcpyHostToDevice);

//     // for calculating energy
//     float* tempArr;
//     cudaMalloc (&tempArr, dim * sizeof (float));

//     // for calculating energy
//     float* tempArr_Host;
//     cudaMallocHost (&tempArr_Host, dim * sizeof (float));

//     cudaBindTexture (0, b_text, b_copy);
//     cudaBindTexture (0, Q_text, Q_copy);

//     for (int n = 0; n < sweeps; n++) {

//         cudaMemcpy (b_copy, b, dim * sizeof (int), cudaMemcpyHostToDevice);

//         slipBinary << <blocks, threads >> > (dim, offset, beta[n], stat, (float)rand ());
//         cudaDeviceSynchronize ();
//         cudaMemcpy (stat_host, stat, 2 * dim * sizeof (float), cudaMemcpyDeviceToHost);

//         // stat[0] = accept, stat[1] = delta_E
//         if (abs (sum (stat_host, dim)) <= 0.1) {
//             offset += offsetIncreasingRate * min (&stat_host[dim], dim);
//             printf ("\toffset added at n=%d\n", n);
//         } else {
//             int index = randChoose (stat_host, dim);
//             b[index] = b[index] * -1 + 1;
//             printf ("\tchosen index =%d\n", index);
//             offset = 0;
//         }

//         // calculate energy ; only needed for testing
//         {
//             cudaMemcpy (b_copy, b, dim * sizeof (int), cudaMemcpyHostToDevice);
//             calculateEnergy << <blocks, threads >> > (b_copy, Q_copy, tempArr, dim);
//             cudaDeviceSynchronize ();
//             cudaMemcpy (tempArr_Host, tempArr, dim * sizeof (float), cudaMemcpyDeviceToHost);
//             energy[n] = sum (tempArr_Host, dim);
//         }
//     }

//     cudaUnbindTexture (b_text);
//     cudaUnbindTexture (Q_text);

//     free (beta);
//     cudaFree (stat);
//     cudaFreeHost (stat_host);
//     cudaFree (b_copy);
//     cudaFree (Q_copy);
//     cudaFree (tempArr);
//     cudaFreeHost (tempArr_Host);
// }

/////////////////////////////////////////////////////////////////////////
/// Below is the code that Pythonf code calls to execute the algorithm ///
/////////////////////////////////////////////////////////////////////////

extern "C" {
    float digitalAnnealingPy (int* b, float* Q, int dim, int sweeps);
}

/**
 * @brief the function that runs the digital annealing algorithm
 *
 * @param b binary array
 * @param Q qubo matrix
 * @param dim dimention of binary array and qubo matrix
 * @param sweeps number of iterations to be done
 */
float digitalAnnealingPy (int* b, float* Q, int dim, int sweeps) {

    // int device;
    // cudaGetDevice (&device);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties (&prop, device);

    int blocks = 32 * 16;
    int threads = dim / blocks + 1;

    float betaStart = 1;
    float betaStop = 500;

    float* beta;
    beta = (float*)malloc (sweeps * sizeof (float));
    getAnnealingBeta (betaStart, betaStop, beta, sweeps);

    float offset = 0;
    float offsetIncreasingRate = 0.1;

    float* stat;
    cudaMalloc (&stat, 2 * dim * sizeof (float));

    float* stat_host;
    cudaMallocHost (&stat_host, 2 * dim * sizeof (float));

    int* b_copy;
    cudaMalloc (&b_copy, dim * sizeof (int));

    float* Q_copy;
    cudaMalloc (&Q_copy, dim * dim * sizeof (float));
    cudaMemcpy (Q_copy, Q, dim * dim * sizeof (float), cudaMemcpyHostToDevice);

    cudaMemcpy (b_copy, b, dim * sizeof (int), cudaMemcpyHostToDevice);

    cudaBindTexture (0, b_text, b_copy);
    cudaBindTexture (0, Q_text, Q_copy);

    // for calculating energy
    float* tempArr;
    cudaMalloc (&tempArr, dim * sizeof (float));

    // for calculating energy
    float* tempArr_Host;
    cudaMallocHost (&tempArr_Host, dim * sizeof (float));

    for (int n = 0; n < sweeps; n++) {

        slipBinary << <blocks, threads >> > (dim, offset, beta[n], stat, (float)rand ());
        cudaDeviceSynchronize ();
        cudaMemcpy (stat_host, stat, 2 * dim * sizeof (float), cudaMemcpyDeviceToHost);



        // printf ("n=%d\n===\n", n);
        // for (int i = 0; i < dim; i++) {
        //     printf ("%d -> %.0f,\t", i, stat_host[dim + i]);
        //     if (i % 7 == 6) { printf ("\n"); }
        // }
        // printf ("\n===\n");
        // stat[0] = accept, stat[1] = delta_E
        if (sum (stat_host, dim) <= 0.001) {
            offset += offsetIncreasingRate * max (&stat_host[dim], dim);
            // printf ("n = %d -> offset added = %.9f, sum = %f\n", n, offset, sum (stat_host, dim));
        } else {
            int index = randChoose (stat_host, dim);
            b[index] = b[index] * -1 + 1;
            offset = 0;
            cudaMemcpy (b_copy, b, dim * sizeof (int), cudaMemcpyHostToDevice);
            // printf ("n = %d -> accepted index = %d, delta_E = %.9f\n", n, index, stat_host[dim + index]);
        }
        if (n % 100 == 0) {
            float energy = 0;
            dot1 << <blocks, threads >> > (tempArr, dim);
            cudaDeviceSynchronize ();
            dot2 << <blocks, threads >> > (tempArr, dim);
            cudaDeviceSynchronize ();
            cudaMemcpy (tempArr_Host, tempArr, dim * sizeof (float), cudaMemcpyDeviceToHost);
            energy = sum (tempArr_Host, dim);
            printf ("\tn = %d --> energy = %.5f\n", n, energy);
        }

    }

    cudaUnbindTexture (b_text);
    cudaUnbindTexture (Q_text);

    free (beta);
    cudaFree (stat);
    cudaFreeHost (stat_host);
    cudaFree (b_copy);
    cudaFree (Q_copy);
    cudaFree (tempArr);
    cudaFreeHost (tempArr_Host);

    // return energy;
    return 0;
}

/////////////////////////////////////////////////////////////////////////
/// Above is the code that Python code calls to execute the algorithm ///
/////////////////////////////////////////////////////////////////////////

// int main () {

//     int dim = 1500;

//     // create a random 40 * 40 array Q
//     // create an inital state([1]) bit array b
//     srand (1);
//     float* Q;
//     int* b;
//     cudaMallocHost (&Q, dim * dim * sizeof (float));
//     cudaMallocHost (&b, dim * sizeof (int));
//     for (int i = 0; i < dim; i++) {
//         b[i] = 1;
//     }
//     for (int i = 0; i < dim * dim; i++) {
//         Q[i] = rand () / ((float)(RAND_MAX - 1) / 2 + 1) - 1;
//     }

//     int sweeps = 100000;
//     float* energy;
//     cudaMallocHost (&energy, sweeps * sizeof (float));

//     digitalAnnealing (b, Q, dim, energy, sweeps);

//     int stride = 1000;
//     for (int i = 0; i < sweeps / stride; i++) {
//         printf ("i=%d --> e=%.5f\n", i * stride, energy[i * stride]);
//     }

//     cudaFree (Q);
//     cudaFree (b);
//     cudaFree (energy);
//     return 0;
// }
