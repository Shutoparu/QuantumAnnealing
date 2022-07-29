#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>

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

    for (int i = 0; i < size-1; i++) {
        if (arr[i] == 1) {
            indicies[nonZeroNum++] = i;
        }
    }
    if (nonZeroNum == 0) {
        return -1;
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
float minNum (float* arr, int size) {
    float min = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
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
 * @param rand a random number
 */
__global__ void checkBinary (int dim, float offset, float beta, float* stat, float rand) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < dim - 1) {
        int flipped = 0;
        float delta_E = 0.0f;

        // check flip
        if (tex1Dfetch (b_text, i) == 0) {
            flipped = 1;
        }

        int idim = i * dim;
        for (int n = 0; n < dim; n++) {
            if (n == i && flipped == 1) {
                delta_E += tex1Dfetch (Q_text, idim + n); // time consuming
            } else {
                delta_E += tex1Dfetch (b_text, n) * tex1Dfetch (Q_text, idim + n); // time consuming
            }
        }

        if (flipped != 0) {
            delta_E = 2 * delta_E - tex1Dfetch (Q_text, idim + i) - offset;
        } else {
            delta_E = -2 * delta_E + tex1Dfetch (Q_text, idim + i) - offset;
        }

        // check energy or check % (check pass)
        float p = exp (-delta_E * beta);
        if (p > float (rand / float (INT_MAX))) {
            stat[i] = 1;
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

//////////////////////////////////////////////////////////////////////////
/// Below is the code that Pythonf code calls to execute the algorithm ///
//////////////////////////////////////////////////////////////////////////

extern "C" {
    void digitalAnnealingPy (int* b, float* Q, int dim, int sweeps, float betaStart, float betaStop, int blocks, int threads);
}

/**
 * @brief the function that runs the digital annealing algorithm
 *
 * @param b binary array
 * @param Q qubo matrix
 * @param dim dimention of binary array and qubo matrix
 * @param sweeps number of iterations to be done
 */
void digitalAnnealingPy (int* b, float* Q, int dim, int sweeps, float betaStart, float betaStop, int blocks, int threads) {

    srand (time (NULL));
    // srand (1);

    float* beta;
    beta = (float*)malloc (sweeps * sizeof (float));
    getAnnealingBeta (betaStart, betaStop, beta, sweeps);

    float offset = 0;
    float offsetIncreasingRate = 0.1;

    float* stat;
    cudaMallocManaged (&stat, 2 * dim * sizeof (float));

    int* b_copy;
    cudaMalloc (&b_copy, dim * sizeof (int));
    cudaMemcpy (b_copy, b, dim * sizeof (int), cudaMemcpyHostToDevice);

    float* Q_copy;
    cudaMalloc (&Q_copy, dim * dim * sizeof (float));
    cudaMemcpy (Q_copy, Q, dim * dim * sizeof (float), cudaMemcpyHostToDevice);

    cudaBindTexture (0, b_text, b_copy);
    cudaBindTexture (0, Q_text, Q_copy);

    // for calculating energy
    // float* tempArr;
    // cudaMalloc (&tempArr, dim * sizeof (float));
    // float* tempArr_Host;
    // cudaMallocHost (&tempArr_Host, dim * sizeof (float));

    for (int n = 0; n < sweeps; n++) {

        checkBinary << <blocks, threads >> > (dim, offset, beta[n], stat, rand ());
        // cudaDeviceSynchronize ();

        int index = randChoose (stat, dim);
        if (index == -1) {
            offset += offsetIncreasingRate * minNum (&stat[dim], dim - 1);
        } else {
            b[index] = b[index] * -1 + 1;
            cudaMemcpy (b_copy, b, dim * sizeof (int), cudaMemcpyHostToDevice);
            offset = 0;
        }

        // if (n % 1000 == 0) {
        //     float energy = 0;
        //     dot1 << <blocks, threads >> > (tempArr, dim);
        //     cudaDeviceSynchronize ();
        //     dot2 << <blocks, threads >> > (tempArr, dim);
        //     cudaDeviceSynchronize ();
        //     cudaMemcpy (tempArr_Host, tempArr, dim * sizeof (float), cudaMemcpyDeviceToHost);
        //     energy = sum (tempArr_Host, dim);
        //     printf ("\tn = %d --> energy = %.5f\n", n, energy);
        // }

    }

    cudaUnbindTexture (b_text);
    cudaUnbindTexture (Q_text);

    free (beta);
    cudaFree (stat);
    cudaFree (b_copy);
    cudaFree (Q_copy);
    // cudaFree (tempArr);
    // cudaFreeHost (tempArr_Host);
}

/////////////////////////////////////////////////////////////////////////
/// Above is the code that Python code calls to execute the algorithm ///
/////////////////////////////////////////////////////////////////////////
