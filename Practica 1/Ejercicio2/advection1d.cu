#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <math.h>

using namespace std;

int index(int i){return  i+1;}
// Blocksize
//#define  BLOCKSIZE 256
//#define  BLOCKSIZE 512
#define  BLOCKSIZE 1024

// Number of mesh points
int n= 60000;



//*************************************************
//Memoria global
// ************************************************
__global__ void FD_kernel1(float * V_in, float * V_out, int n)
  {
    int i=threadIdx.x+blockDim.x*blockIdx.x+2;

    if (i<n+4)
      V_out[i]=(pow(V_in[i-2], 5)+2*pow(V_in[i-1], 5)+pow(V_in[i], 5)-3*pow(V_in[i+1], 5)+5*pow(V_in[i+2], 5))/24;

    // Boundary Conditions
    //si es primer thread, ponemos valor 0 a las dos primeras posiciones del vector
    if (i==1){
      V_out[0]=0;
      V_out[1]=0;
    }
    // ponemos valor 0 a las dos ultimas posiciones del vector.
    if (i==n+1){
      V_out[n+3]=0;
      V_out[n+2]=0;
    }
  }



//*************************************************
// Memoria compartida
// ************************************************
__global__ void FD_kernel2(float * V_in, float * V_out, int n)
  {
    int li=threadIdx.x+2;   //local index in shared memory vector
    int gi=   blockDim.x*blockIdx.x+threadIdx.x+2; // global memory index
    __shared__ float s_v[BLOCKSIZE + 4];  //shared mem. vector
    float result;

   // Load Tile in shared memory
    if (gi<n+4) s_v[li]=V_in[gi];
  __syncthreads();

   if (gi<n+4)
    {
     result=(pow(s_v[li-2], 5)+2*pow(s_v[li-1], 5)+pow(s_v[li], 5)-3*pow(s_v[li+1], 5)+5*pow(s_v[li+2], 5))/24;
     V_out[gi]=result;
    }

   // Boundary Conditions
    if (gi==1) {
      V_out[0]=0;
      V_out[1]=0;
    }
    if (gi==n+1){
      V_out[n+2]=0;
      V_out[n+3]=0;
    }

  }



//******************************
//**** MAIN FUNCTION ***********

int main(int argc, char* argv[]){

//******************************
   //Get GPU information
    int devID;
    cudaDeviceProp props;
    cudaError_t err;
    err=cudaGetDevice(&devID);
    if (err!=cudaSuccess) {cout<<"ERRORRR"<<endl;}
    cudaGetDeviceProperties(&props, devID);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);


cout<<"Introduce number of points (1000-200000)"<<endl;
cin>>n;


  //variable para version memoria global
  float * vector    =new float[n+5];
  float * vector_new=new float[n+5];
  float * h_out=new float[n+5];

  int size=(n+5)*sizeof(float);

  // Allocation in device mem. for d_in
  float * d_in=NULL;
  err=cudaMalloc((void **) &d_in, size);
  if (err!=cudaSuccess) {cout<<"ALLOCATION ERROR"<<endl;}
 // Allocation in device mem. for d_out
  float * d_out=NULL;
  err=cudaMalloc((void **) &d_out, size);
  if (err!=cudaSuccess) {cout<<"ALLOCATION ERROR"<<endl;}

  //variable para version memoria compartida
  float * h_out_share=new float[n+5];

  // Allocation in device mem. for d_in
  float * d_in_share=NULL;
  err=cudaMalloc((void **) &d_in_share, size);
  if (err!=cudaSuccess) {cout<<"ALLOCATION ERROR"<<endl;}
 // Allocation in device mem. for d_out
  float * d_out_share=NULL;
  err=cudaMalloc((void **) &d_out_share, size);
  if (err!=cudaSuccess) {cout<<"ALLOCATION ERROR"<<endl;}



  // Initial values
  for(int i=0;i<=n;i++)
   {
   vector[index(i)]=i;
   }

   // Impose Boundary Conditions
      vector[index(-1)] =0;
      vector[index(-2)] =0;
      vector[index(n+1)]=0;
      vector[index(n+2)]=0;
//**************************
// GPU phase memoria global
//**************************

 // Take initial time
 double  t1_global=clock();

 // Copy values to device memory
 err=cudaMemcpy(d_in,vector,size,cudaMemcpyHostToDevice);

 if (err!=cudaSuccess) {cout<<"GPU COPY ERROR"<<endl;}

 int blocksPerGrid_global =(int) ceil((float)(n+4)/BLOCKSIZE);

 // ********* Kernel Launch ************************************
 FD_kernel1<<<blocksPerGrid_global, BLOCKSIZE >>> (d_in, d_out, n);
 // ************************************************************

 err = cudaGetLastError();
 if (err != cudaSuccess)
 {
   fprintf(stderr, "Failed to launch kernel! %d \n",err);
   exit(EXIT_FAILURE);
 }

 cudaDeviceSynchronize();

 cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);


 double Tgpu=clock();
 Tgpu=(Tgpu-t1_global)/CLOCKS_PER_SEC;

/*
 for (int i=0; i<n+4; i++)
 {
   printf(" vector resultado[%d]=%f\n",i,h_out[i]);
 }
*/
 float max_gpu=h_out[0];
 for(int i=0;i<n+4;i++){
   float auxg=h_out[i];
   if(auxg>max_gpu){
     max_gpu=auxg;
   }
 }



 //**************************
 // GPU phase memoria compartida
 //**************************

  // Take initial time
  double  t1_share=clock();

  // Copy values to device memory
  err=cudaMemcpy(d_in_share,vector,size,cudaMemcpyHostToDevice);

  if (err!=cudaSuccess) {cout<<"GPU COPY ERROR"<<endl;}

  int blocksPerGrid_share =(int) ceil((float)(n+4)/BLOCKSIZE);

  // ********* Kernel Launch ************************************
  FD_kernel2<<<blocksPerGrid_share, BLOCKSIZE >>> (d_in_share, d_out_share, n);
  // ************************************************************

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch kernel! %d \n",err);
    exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();

  cudaMemcpy(h_out_share, d_out_share, size, cudaMemcpyDeviceToHost);


  double Tgpu_share=clock();
  Tgpu_share=(Tgpu_share-t1_share)/CLOCKS_PER_SEC;

/*
  for (int i=0; i<n+4; i++)
  {
    printf(" vector_share resultado [%d]=%f\n",i,h_out[i]);
  }*/

  float max_gpu_share=h_out[0];
  for(int i=0;i<n+4;i++){
    float auxgs=h_out[i];
    if(auxgs>max_gpu_share){
      max_gpu_share=auxgs;
    }
  }





//**************************
// CPU phase
//**************************

double  t1cpu=clock();

for(int i=0;i<n+4;i++){
  vector_new[i]=(pow(vector[i-2], 5)+2*pow(vector[i-1], 5)+pow(vector[i], 5)-3*pow(vector[i+1], 5)+5*pow(vector[i+2], 5))/24;
}

vector_new[index(-1)] =0;
vector_new[index(-2)] =0;
vector_new[index(n+1)]=0;
vector_new[index(n+2)]=0;
/*
for (int i=0; i<n+4; i++)
{
  printf(" vector new[%d]=%f\n",i,vector_new[i]);
}
*/
float max_cpu=vector_new[0];
for(int i=0;i<n+4;i++){
  float aux=vector_new[i];
  if(aux>max_cpu){
    max_cpu=aux;
  }
}

double Tcpu=clock();
Tcpu=(Tcpu-t1cpu)/CLOCKS_PER_SEC;


cout<<endl;
cout<< "GPU global Time= "<<Tgpu<<endl<<endl;
cout<< "GPU compartida Time= "<<Tgpu_share<<endl<<endl;
cout<< "CPU Time= "<<Tcpu<<endl<<endl;

cout<<"maximo en GPU global:" << max_gpu<<endl;
cout<<"maximo en GPU compartida:" << max_gpu_share<<endl;
cout<< "Max en CPU= "<<max_cpu<<endl<<endl;


 return 0;
}
