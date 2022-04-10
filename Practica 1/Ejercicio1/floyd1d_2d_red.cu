#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "stdio.h"
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

//#define blocksize 64
//#define blocksize 256
#define blocksize 1024

//#define blocksize2D 8
//#define blocksize2D 16
#define blocksize2D 32

//#define blocksizeRed 64
//#define blocksizeRed 256
#define blocksizeRed 1024

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//****************************Floyd 1D**********************************************
__global__ void floyd_kernel(int * M, const int nverts, const int k) {
    int ij = threadIdx.x + blockDim.x * blockIdx.x;
    const int i= ij / nverts;
    const int j= ij - i * nverts;
    if (i< nverts && j< nverts) {
    int Mij = M[ij];
    if (i != j && i != k && j != k) {
        int Mikj = M[i * nverts + k] + M[k * nverts + j];
        Mij = (Mij > Mikj) ? Mikj : Mij;
        M[ij] = Mij;
                }
  }
}



//****************************Floyd 2D**********************************************
__global__ void floyd2d_kernel(int * M, const int nverts, const int k) {
		//tomando como ejemplo el algoritmo de la suma de matriz 2d, calculamos la fila y la columna de la matriz.
		// fila de la matriz
		int j = blockIdx.x * blockDim.x + threadIdx.x;
		// columna de la matriz
		int i = blockIdx.y * blockDim.y + threadIdx.y;
		// calculamos ahora indice de la matriz segun la fila y columna
		int ij = i * nverts + j;

    if (i< nverts && j< nverts) {
    int Mij = M[ij];
    if (i != j && i != k && j != k) {
        int Mikj = M[i * nverts + k] + M[k * nverts + j];
        Mij = (Mij > Mikj) ? Mikj : Mij;
        M[ij] = Mij;
                }
  }
}


//****************************Reduccion Max**********************************************
//toma como entrada la salida del algoritmo floyd
__global__ void reduceMax(int * V_in, int * V_out, const int N) {
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? V_in[i] : -100000000.0f);
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){
	  if (tid < s)
		if(sdata[tid] > sdata[tid+s])
                    sdata[tid] = sdata[tid+s];
	  __syncthreads();
	}
	if (tid == 0)
           V_out[blockIdx.x] = sdata[0];
}


int main (int argc, char *argv[]) {

	if (argc != 2) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}


  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if(err != cudaSuccess) {
		cout << "ERRORRR" << endl;
	}


cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	Graph G;
	G.lee(argv[1]);// Read the Graph

	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;

	const int nverts2 = nverts * nverts;
	int size = nverts2*sizeof(int);

	//variables para floyd1D
	int *c_Out_M = new int[nverts2];
	int *d_In_M = NULL;

	//variables para floyd2D
	int *c_Out_M_2D = new int[nverts2];
	int *d_In_M_2D = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	err = cudaMalloc((void **) &d_In_M_2D, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	int *A = G.Get_Matrix();

//****************************GPU phase 1D**********************************************
	double  t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	int threadsPerBlock = blocksize;
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;

	  floyd_kernel<<<blocksPerGrid,threadsPerBlock >>>(d_In_M, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
	  	fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu = cpuSecond()-t1;

	cout << "Tiempo gastado GPU floyd1d= " << Tgpu << endl << endl;

//****************************GPU phase 2D**********************************************

	t1 = cpuSecond();

	err = cudaMemcpy(d_In_M_2D, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
		dim3 threadsPerBlock2D(blocksize2D, blocksize2D);
		dim3 blocksPerGrid2D (ceil ((float)(nverts)/threadsPerBlock2D.x),
											  ceil ((float)(nverts)/threadsPerBlock2D.y) );

		floyd2d_kernel<<<blocksPerGrid2D,threadsPerBlock2D >>>(d_In_M_2D, nverts, k);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
			exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M_2D, d_In_M_2D, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu2 = cpuSecond()-t1;

	cout << "Tiempo gastado GPU floyd2D= " << Tgpu2 << endl << endl;


//****************************CPU phase **********************************************
	t1 = cpuSecond();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
	       			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       }
	   }
	}

  double t2 = cpuSecond() - t1;
  cout << "Tiempo gastado CPU= " << t2 << endl << endl;
  cout << "Ganancia floyd1D = " << t2 / Tgpu << endl;
	cout << "Ganancia floyd2D = " << t2 / Tgpu2 << endl;


  for(int i = 0; i < nverts; i++)
    for(int j = 0;j < nverts; j++)
       if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
         cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;


//****************************Reduccion Max**********************************************

//variables de Reduccion

//resultado en host
	int *c_Out_M_Red = new int[size];
//entrada de device
	int *d_In_M_Red = NULL;
//salida de host
	int *d_out_M_Red = NULL;


// Reservar memoria
	err = cudaMalloc((void **) &d_In_M_Red, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	err = cudaMalloc((void **) &d_out_M_Red, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

//copiar resultado del floyd2d en el d_In_M_Red
/*	err = cudaMemcpy(d_In_M_Red, c_Out_M_2D, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}*/

	err = cudaMemcpy(d_In_M_Red, d_In_M_2D, size, cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}



	dim3 threadsPerBlockRed(blocksizeRed,1);
	dim3 numBlocksRed(ceil ((int)(nverts)/threadsPerBlockRed.x), 1);

	reduceMax<<<numBlocksRed, threadsPerBlockRed>>>(d_In_M_Red, d_out_M_Red, nverts,numBlocksRed.x*sizeof(int));

	cudaMemcpy(c_Out_M_Red, d_out_M_Red, numBlocksRed.x*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	int max_cpu = A[0];
	for (int i = 0; i < nverts; ++i){
		for (int j = 0; j < nverts; ++j){
			int auxcpu = A[i * nverts + j];
			//printf(" A[%d][%d]=%d\n",i,j,A[i*nverts+j]);
			if(auxcpu>max_cpu){
				max_cpu=auxcpu;
			}
		}
	}

	int max_gpu=c_Out_M_Red[0];
	for (int i=0; i<nverts;i++){
	  int auxgpu = c_Out_M_Red[i];
		//printf(" c_Out_M_Red[%d]=%d\n",i,c_Out_M_Red[i]);
		if(auxgpu > max_gpu){
			max_gpu=auxgpu;
		}
	}

	cout<<" Max on GPU ="<<max_gpu<<"  Max on CPU="<< max_cpu<<endl;

		/* Free the memory */
		free(c_Out_M_Red);free(c_Out_M); free(c_Out_M_2D);
		cudaFree(d_In_M_Red); cudaFree(d_In_M); cudaFree(d_In_M_2D); cudaFree(d_out_M_Red);

}
