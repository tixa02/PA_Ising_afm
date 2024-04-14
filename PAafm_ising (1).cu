//
// PAising version 1.16. This program employs standard spin coding.
// This program is introduced in the paper:
// L.Yu. Barash, M. Weigel, M. Borovsky, W. Janke, L.N. Shchur, GPU accelerated population annealing algorithm
// This program is licensed under a Creative Commons Attribution 4.0 International License:
// http://creativecommons.org/licenses/by/4.0/
//
// Use command line option -? to print list of available command line options.
// All of the command line options are optional.
//

#include <iostream>
#include <fstream>
#include <iomanip>
#include <curand_kernel.h>
#include <string>
#ifdef _WIN32			// this program is compatible with any of the Windows, Unix/Linux, MacOS environments
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// #define MHR			// uncomment/comment to enable/disable multi-histogram reweighting
// #define AdaptiveStep		// uncomment/comment to enable/disable adaptive temperature step
// #define EnergiesPopStore	// uncomment/comment to enable/disable storing energies at each T
// #define LatticePopStore	// uncomment/comment to enable/disable storing lattice at each T

#define Ldiv3   (44)
#define L	(Ldiv3*3)		// linear size of the system in x,y direction
#define Ndiv9 (Ldiv3*Ldiv3)
#define N       (L*L)


#define J1 1.0
#define J2 0.15
#define J3 0.1




#define RNGseed	467//time(NULL)	// Use 32-bit integer as a seed for random number generation, e.g., time(NULL)

typedef curandStatePhilox4_32_10_t RNGState;

unsigned int EQsweeps = 50;				// number of equilibration sweeps

double Binit = 0;	//10				// initial inverse temperature
double Bfin = 10;					// final inverse temperature
double dBinit = 0.005;                             	// inverse temperature step

#ifdef AdaptiveStep
double MinOverlap = 0.85;			// minimal value of acceptable overlap of energy histograms
	double MaxOverlap = 0.87;			// maximal value of acceptable overlap of energy histograms
#endif

int Rinit = 20000;					// Initial size of population of replicas

int runs = 1;						// number of population annealing algorithm independent runs

int OutputPrecision = 11;				// precision (number of digits) of the output

#ifdef MHR
const short MHR_Niter = 1;	// number of iterations for multi-histogram analysis (single iteration is usually sufficient)
#endif

const int boltzTableL = 686;				// Boltzmann factor table length
const int nBmax = 10000;                                // number of temperature steps should not exceed nBmax

texture<int2,1,cudaReadModeElementType> boltzT;
int* sublat;//new sublat
using namespace std;

#define EQthreads 128	// number of threads per block for the equilibration kernel
#define Nthreads  1024	// number of threads per block for the parallel reduction algorithm
// Use Nthreads=1024 for CUDA compute capability 2.0 and above; Nthreads=512 for old devices with CUDA compute capability 1.x.

double* Qd; double* ioverlapd;

// struct Replica covers all information about the replica including its configuration, sublattice magnetizations,
// internal energy and number of replica's offspring
struct Replica{
    signed char g[N];
    			signed char g2[N];// sublattice configurations
    double IE;					// internal energy
    int M; //magnetization
    union{ double ValDouble; unsigned int ValInt[2]; } ParSum; // is used for storing sums
    unsigned int Roff;
    double p;
    int cx, cy;
    int i,s0,s1,s2,s3;
    int row[6], col[6];
    int j1[6], j2[6], j3[6], x1[6], y1[6], x2[6], y2[6], i2[6];
    signed char operator()(int i, int j){
        return g[(i/3)*Ldiv3 + j/3 +(j%3 + 3*(i%3)) *N/9];
    }
};

void print_replica(Replica& R, int r){
//     cout << "run = " << r << std::endl;
//     printf("cx=%d, cy=%d\n", R.cx, R.cy);
//     cout << "p = " << R.p << std::endl;
//     cout << "s0 = " << R.s0 << std::endl;
//     cout << "s1 = " << R.s1 << std::endl;
//     cout << "s2 = " << R.s2 << std::endl;
//     cout << "s3 = " << R.s3 << std::endl;
//     cout << "i = " << R.i << std::endl;
//     cout << "row:";
//     for (int j=0; j<6; j++){
//         cout << ((R.row[j] == 1) ? "+ ":"- ");
//     }
//     cout << "col:";
//     for (int j=0; j<6; j++){
//         cout << ((R.col[j] == 1) ? "+ ":"- ");
//     }
//     printf("x: l: %d, ul: %d, u: %d, r: %d, dr:%d d %d\n", R.x1[0], R.x1[1], R.x1[2], R.x1[3], R.x1[4], R.x1[5]);
//     printf("y: l: %d, ul: %d, u: %d, r: %d, dr:%d d %d\n", R.y1[0], R.y1[1], R.y1[2], R.y1[3], R.y1[4], R.y1[5]);
//     printf("l: %d, ul: %d, u: %d, r: %d, dr:%d d %d\n", R.j1[0], R.j1[1], R.j1[2], R.j1[3], R.j1[4], R.j1[5]);
//     printf("ul l: %d, ul u: %d, ur: %d, dr r: %d, dr d:%d dl %d\n", R.j2[0], R.j2[1], R.j2[2], R.j2[3], R.j2[4], R.j2[5]);
//     printf("l: %d, ul: %d, u: %d, r: %d, dr :%d d %d\n", R.j3[0], R.j3[1], R.j3[2], R.j3[3], R.j3[4], R.j3[5]);
     cout << "Energy of replica = " << R.IE << std::endl;
    for (int i=0; i<L; ++i){
        cout << string(L-i, ' ');
        for (int j = 0; j<L; ++j){
            if (0){//(i==64 && j ==9){
            cout << "! ";
            }
            else
            cout << ((R(i,j) == 1) ? "+ ":"- ");
        }
        cout << endl;
    }
    cout << endl;
}

// CUDA error checking macro
#define CUDAErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s ; %s ; line %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static __inline__ __device__ double fetch_double(texture<int2,1> t, int i) // texture fetching for double precision floats
{
    int2 v = tex1Dfetch(t,i);
    return __hiloint2double(v.y, v.x);
}

template <class sometype> __inline__ __device__ sometype smallblockReduceSum(sometype val) // use when blockDim.x < 32
{											   // blockDim.x must be a power of 2
    static __shared__ sometype shared[32];
    shared[threadIdx.x] = val;
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        __syncthreads(); if (threadIdx.x < stride)  shared[threadIdx.x] += shared[threadIdx.x+stride];
    }
    __syncthreads(); return shared[0];
}

#if  (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300)
template <class sometype> __inline__ __device__ sometype warpReduceSum(sometype val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2) val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

template <class sometype> __inline__ __device__ sometype blockReduceSum(sometype val)	 // use when blockDim.x is divisible by 32
{
	static __shared__ sometype shared[32];			// one needs to additionally synchronize threads after execution
	int lane = threadIdx.x % warpSize;			// in the case of multiple use of blockReduceSum in a single kernel
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum(val);
	if (lane==0) shared[wid]=val;
	__syncthreads();
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
	if (wid==0) val = warpReduceSum(val);
	return val;
}
#else
template <class sometype> __inline__ __device__ sometype blockReduceSum(sometype val)	// blockDim.x must be a power of 2
{
    static __shared__ sometype shared[Nthreads];
    shared[threadIdx.x] = val;
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        __syncthreads(); if (threadIdx.x < stride)  shared[threadIdx.x] += shared[threadIdx.x+stride];
    }
    __syncthreads(); return shared[0];
}
#endif

#if (__CUDACC_VER_MAJOR__ < 8) || ( defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 )
__device__ double atomicAdd(double* address, double val) // allows to use atomicAdd operation for double precision floating point values
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void ReplicaInit(Replica* Rd, unsigned long long rng_seed, unsigned long long initial_sequence){ // initialization of spin lattices of all replicas
    unsigned int B = blockIdx.x; unsigned int t = threadIdx.x;
    RNGState localrng; curand_init(rng_seed,initial_sequence+(t+B*EQthreads),0,&localrng);
    for (unsigned int idx = t; idx < (N); idx += EQthreads){
        Rd[B].g[idx] = ( curand_uniform_double(&localrng) >= 0.5) ? 1 : -1;
        //Rd[B].gB[idx] = ( curand_uniform_double(&localrng) >= 0.5) ? 1 : -1;
    }
}

__global__ void checkKerALL(Replica* Rd, unsigned int sweeps, unsigned long long rng_seed,
                            unsigned long long initial_sequence, int* sublat) // equilibration process
{
    signed char idxT0, idxT1, idxT2, idxT3;
    unsigned int t = threadIdx.x, B = blockIdx.x, idx;
    unsigned int cx, cy;	// B is replica index
    unsigned int j1x[6];
    unsigned int j1y[6];
    unsigned int j2x[6];
    unsigned int j2y[6];
    unsigned int j3x[6];
    unsigned int j3y[6];
    RNGState localrng; curand_init(rng_seed,initial_sequence+(t+blockIdx.x*EQthreads),0,&localrng);



    for(int sweep=0; sweep<sweeps; sweep++){ // sweeps loop

        unsigned int s1, s2, s3;
        for (int i = 0; i<9; ++i){//sublattice i
            for (idx = t; idx < (Ndiv9); idx += EQthreads){
                cx = sublat[2*(idx + i*Ndiv9)];
                cy = sublat[1+2*(idx + i*Ndiv9)];

                //first order neighbours
                j1x[0]=(cx-1+L)%L; j1y[0]=cy;//left
                j1x[1]=j1x[0];     j1y[1]=(cy-1+L)%L;//upper-left
                j1x[2]=cx;         j1y[2]=j1y[1];//upper
                j1x[3]=(cx+1)%L;   j1y[3]=cy;//right
                j1x[4]=j1x[3];     j1y[4]=(cy+1)%L;//lower-right
                j1x[5]=cx;         j1y[5]=j1y[4];//lower

                //third order neighbours
                j3x[0]=(cx-2+L)%L; j3y[0]=cy;//left left
                j3x[1]=j3x[0];     j3y[1]=(cy-2+L)%L;//upper-left upper-left
                j3x[2]=cx;         j3y[2]=j3y[1];//upper upper
                j3x[3]=(cx+2)%L;   j3y[3]=cy;//right right
                j3x[4]=j3x[3];     j3y[4]=(cy+2)%L;//lower-right lower-right
                j3x[5]=cx;         j3y[5]=j3y[4];//lower lower

                //second order neighbours
                j2x[0]=j3x[0];     j2y[0]=j1y[2];//upper-left left
                j2x[1]=j1x[0];     j2y[1]=j3y[2];//upper-left upper
                j2x[2]=j1x[3];     j2y[2]=j1y[2];//upper right
                j2x[3]=j3x[3];     j2y[3]=j1y[5];//lower-right right
                j2x[4]=j1x[3];     j2y[4]=j3y[5];//lower-right lower
                j2x[5]=j1x[0];     j2y[5]=j1y[5];//left lower



                s1=0;s2=0;s3=0;
                for (int j=0; j<6; ++j){
                    s1+= Rd[B].g[(j1y[j]/3)*Ldiv3 + j1x[j]/3 +(j1x[j]%3 + 3*(j1y[j]%3)) * Ndiv9];
                    s2+= Rd[B].g[(j2y[j]/3)*Ldiv3 + j2x[j]/3 +(j2x[j]%3 + 3*(j2y[j]%3)) * Ndiv9];
                    s3+= Rd[B].g[(j3y[j]/3)*Ldiv3 + j3x[j]/3 +(j3x[j]%3 + 3*(j3y[j]%3)) * Ndiv9];
                }



                idxT0 = (1 +  Rd[B].g[idx + i*Ndiv9])/2;
                idxT1 = (6 + s1)/2;
                idxT2 = (6 + s2)/2;
                idxT3 = (6 + s3)/2;

                if (cx==9 && cy==64 && sweep==(sweeps-1)){
                    for (int j=0; j<L; ++j){
                        for (int k=0; k<L; ++k){
                            Rd[B].g2[(j/3)*Ldiv3 + k/3 +(k%3 + 3*(j%3)) *N/9]=Rd[B].g[(j/3)*Ldiv3 + k/3 +(k%3 + 3*(j%3)) * Ndiv9];
                        }
                    }
                    for (int j=0; j<6; ++j){
                        Rd[B].row[j]=Rd[B].g[(0/3)*Ldiv3 + j/3 +(j%3 + 3*(0%3)) * Ndiv9];
                        Rd[B].col[j]=Rd[B].g[(j/3)*Ldiv3 + 0/3 +(0%3 + 3*(j%3)) * Ndiv9];
                        Rd[B].x1[j] = j1x[j];
                        Rd[B].y1[j] = j1y[j];
                        Rd[B].x2[j] = j2x[j];
                        Rd[B].y2[j] = j2y[j];
                        Rd[B].i2[j]=(j2y[j]/3)*Ldiv3 + j2x[j]/3 +(j2x[j]%3 + 3*(j2y[j]%3)) * Ndiv9;
                        //Rd[B].i3[j]=(j3y[j]/3)*Ldiv3 + j3x[j]/3 +(j3x[j]%3 + 3*(j3y[j]%3)) * Ndiv9;
                        Rd[B].j1[j]=Rd[B].g[(j1y[j]/3)*Ldiv3 + j1x[j]/3 +(j1x[j]%3 + 3*(j1y[j]%3)) * Ndiv9];
                        Rd[B].j2[j]=Rd[B].g[(j2y[j]/3)*Ldiv3 + j2x[j]/3 +(j2x[j]%3 + 3*(j2y[j]%3)) * Ndiv9];
                        Rd[B].j3[j]=Rd[B].g[(j3y[j]/3)*Ldiv3 + j3x[j]/3 +(j3x[j]%3 + 3*(j3y[j]%3)) * Ndiv9];
                    }
                    Rd[B].cx = cx;
                    Rd[B].cy = cy;
                    Rd[B].p=fetch_double(boltzT, idxT0 + 2*(idxT1*49 + 7*idxT2 + idxT3)) ;
                    Rd[B].s0=Rd[B].g[idx + i*Ndiv9];
                    Rd[B].s1=s1;
                    Rd[B].s2=s2;
                    Rd[B].s3=s3;
                    Rd[B].i = idxT0 + 2*(idxT1*49 + 7*idxT2 + idxT3);
                }

                if( curand_uniform_double(&localrng) < fetch_double(boltzT, idxT0 + 2*(idxT1*49 + 7*idxT2 + idxT3)) ) Rd[B].g[idx + i*Ndiv9] *= -1;
            }
            __syncthreads();
        }
    }
}


__global__ void energyKer(Replica* Rd) // calculation of energy and magnetization for each replica
{
    signed char s[9];
    double e_sub[9];
    signed char i2[9];
    signed char i4[9];
    signed char i6[9];
    signed char i2_2[9];
    signed char i4_2[9];
    signed char i6_2[9];
    signed char i2_3[9];
    signed char i4_3[9];
    signed char i6_3[9];
    unsigned int idx, t = threadIdx.x, B = blockIdx.x, iL, iU, iD, iUL, iL_D;
    unsigned int tx0, ty0, tyL, txL;
    double e, m = 0;
    double Energy = 0, Magnetization = 0;
    for (idx = t; idx < (Ndiv9); idx += EQthreads){
        if(t < EQthreads){
            for (int i=0; i<9; ++i){
                s[i] = Rd[B].g[idx + i*Ndiv9];
            }
            ty0 = idx / Ldiv3; tx0 = idx - ty0*Ldiv3;

            //first order neighbours
            iL = ty0 * Ldiv3 + (tx0 + Ldiv3 - 1) % Ldiv3;
            tyL = iL / Ldiv3; txL = iL - tyL*Ldiv3;
            iD = ((ty0 + 1) % Ldiv3) * Ldiv3 + tx0;
            iU = ((ty0 + Ldiv3 - 1) % Ldiv3) * Ldiv3 + tx0;
            iUL = ((tyL + Ldiv3 - 1) % Ldiv3) * Ldiv3 + txL; // iU(iL)
            //left
            i2[0]=Rd[B].g[iL + 2*Ndiv9];
            i2[1]=s[0];
            i2[2]=s[1];
            i2[3]=Rd[B].g[iL + 5*Ndiv9];
            i2[4]=s[3];
            i2[5]=s[4];
            i2[6]=Rd[B].g[iL + 8*Ndiv9];
            i2[7]=s[6];
            i2[8]=s[7];

            //upper
            i4[0]=Rd[B].g[iU + 6*Ndiv9];
            i4[1]=Rd[B].g[iU + 7*Ndiv9];
            i4[2]=Rd[B].g[iU + 8*Ndiv9];
            i4[3]=s[0];
            i4[4]=s[1];
            i4[5]=s[2];
            i4[6]=s[3];
            i4[7]=s[4];
            i4[8]=s[5];

            //upper left
            i6[0]=Rd[B].g[iUL + 8*Ndiv9];
            i6[1]=Rd[B].g[iU + 6*Ndiv9];
            i6[2]=Rd[B].g[iU + 7*Ndiv9];
            i6[3]=Rd[B].g[iL + 2*Ndiv9];
            i6[4]=s[0];
            i6[5]=s[1];
            i6[6]=Rd[B].g[iL + 5*Ndiv9];
            i6[7]=s[3];
            i6[8]=s[4];

            //second order neighbours

            iL_D=((tyL + 1) % Ldiv3) * Ldiv3 + txL;//iD(iL)
            // UL U
            i2_2[0]=Rd[B].g[iUL + 5*Ndiv9];
            i2_2[1]=Rd[B].g[iU + 3*Ndiv9];
            i2_2[2]=Rd[B].g[iU + 4*Ndiv9];
            i2_2[3]=Rd[B].g[iUL + 8*Ndiv9];
            i2_2[4]=Rd[B].g[iU + 6*Ndiv9];
            i2_2[5]=Rd[B].g[iUL + 7*Ndiv9];
            i2_2[6]=Rd[B].g[iL + 2*Ndiv9];
            i2_2[7]=s[0];
            i2_2[8]=s[1];
            // UL L
            i4_2[0]=Rd[B].g[iUL + 7*Ndiv9];
            i4_2[1]=Rd[B].g[iUL + 8*Ndiv9];
            i4_2[2]=Rd[B].g[iU + 6*Ndiv9];
            i4_2[3]=Rd[B].g[iL + Ndiv9];
            i4_2[4]=Rd[B].g[iL + 2*Ndiv9];
            i4_2[5]=s[0];
            i4_2[6]=Rd[B].g[iL + 4*Ndiv9];
            i4_2[7]=Rd[B].g[iL + 5*Ndiv9];
            i4_2[8]=s[3];
            // L D
            i6_2[0]=Rd[B].g[iL + 5*Ndiv9];
            i6_2[1]=s[3];
            i6_2[2]=s[4];
            i6_2[3]=Rd[B].g[iL + 8*Ndiv9];
            i6_2[4]=s[6];
            i6_2[5]=s[7];
            i6_2[6]=Rd[B].g[iL_D + 2*Ndiv9];
            i6_2[7]=Rd[B].g[iD];
            i6_2[8]=Rd[B].g[iD + Ndiv9];
            //third order neighbours

            //U U
            i2_3[0]=Rd[B].g[iU + 3*Ndiv9];
            i2_3[1]=Rd[B].g[iU + 4*Ndiv9];
            i2_3[2]=Rd[B].g[iU + 5*Ndiv9];
            i2_3[3]=Rd[B].g[iU + 6*Ndiv9];
            i2_3[4]=Rd[B].g[iU + 7*Ndiv9];
            i2_3[5]=Rd[B].g[iU + 8*Ndiv9];
            i2_3[6]=s[0];
            i2_3[7]=s[1];
            i2_3[8]=s[2];
            //L L
            i4_3[0]=Rd[B].g[iL + Ndiv9];
            i4_3[1]=Rd[B].g[iL + 2*Ndiv9];
            i4_3[2]=s[0];
            i4_3[3]=Rd[B].g[iL + 4*Ndiv9];
            i4_3[4]=Rd[B].g[iL + 5*Ndiv9];
            i4_3[5]=s[3];
            i4_3[6]=Rd[B].g[iL + 7*Ndiv9];
            i4_3[7]=Rd[B].g[iL + 8*Ndiv9];
            i4_3[8]=s[6];
            //UL UL
            i6_3[0]=Rd[B].g[iUL + 4*Ndiv9];
            i6_3[1]=Rd[B].g[iUL + 5*Ndiv9];
            i6_3[2]=Rd[B].g[iU + 3*Ndiv9];
            i6_3[3]=Rd[B].g[iUL + 7*Ndiv9];
            i6_3[4]=Rd[B].g[iUL + 8*Ndiv9];
            i6_3[5]=Rd[B].g[iU + 6*Ndiv9];
            i6_3[6]=Rd[B].g[iL + Ndiv9];
            i6_3[7]=Rd[B].g[iL + 2*Ndiv9];
            i6_3[8]=s[0];
            //
            e=0;
            for (int i=0; i<9; ++i){
                e_sub[i] = s[i]*(J1*(i2[i] + i4[i] + i6[i]) + \
                 J2*(i2_2[i] + i4_2[i] + i6_2[i]) + J3*(i2_3[i] + i4_3[i] + i6_3[i]));
                e += e_sub[i];
            }
        } else e = m = 0;
        e = blockReduceSum<double>(e); __syncthreads();
        m = blockReduceSum<int>(m); __syncthreads();
        if (t == 0) { Energy += e; Magnetization += m; }
    }
    if ( t == 0 ) { Rd[B].IE = Energy; Rd[B].M = Magnetization; }
}

__global__ void QKer(Replica* Rd, int R, double dB, double Emean, int CalcPart, double* Qd)  // calculation of partition function ratio
{
    double factor;				// summation of exponential Boltzmann-like factors in deterministic order
    if(CalcPart == 0){                      // first part of the calculation
        int t = threadIdx.x; int b = blockIdx.x; int idx = t + Nthreads * b;
        factor = (idx < R) ? exp(-dB*(Rd[idx].IE-Emean)) : 0.0;
        factor = blockReduceSum<double>(factor);
        if(t == 0) Rd[idx].ParSum.ValDouble = factor; // sum for all threads in current block is saved to global memory
    } else{					// second part of the calculation, summation of the partial sums
        double MyParSum = 0; int j = 0, t = threadIdx.x;
        for (j = 0; j*Nthreads < R; j+=Nthreads){
            factor = (t+j)*Nthreads < R ? Rd[(t+j)*Nthreads].ParSum.ValDouble : 0.0;
            factor = blockReduceSum<double>(factor); __syncthreads();
            MyParSum += factor;
        }
        if(t==0) *Qd = MyParSum;
    }
}

__global__ void CalcTauKer(Replica* Rd, int Rinit, int R, double lnQ, double dB, unsigned int* Rnew, unsigned long long rng_seed, unsigned long long initial_sequence) // calculation of numbers of copies for all replicas
{
    unsigned int parS;
    int t = threadIdx.x; int b = blockIdx.x;
    int idx = t + Nthreads * b; double mu, mufloor;
    RNGState localrng; curand_init(rng_seed,initial_sequence+idx,0,&localrng);
    if (idx < R){		// nearest integer resampling
        mu = ((double)Rinit)/R*exp(-dB*(double)Rd[idx].IE - lnQ);
        mufloor = floor(mu);
        if(curand_uniform_double(&localrng) < (mu-mufloor))
            parS = Rd[idx].Roff = mufloor + 1;
        else    parS = Rd[idx].Roff = mufloor;	// number of copies
    } else parS = 0;
    parS = blockReduceSum<unsigned int>(parS);
    if(t==0){
        Rd[idx].ParSum.ValInt[1] = parS;  // sum of Roff for all threads in current block
        atomicAdd(Rnew,parS); // we save new population size
    }
}

__global__ void CalcParSum(Replica* Repd, int R) // calculation of {sum_{j=0}^i Roff} for each replica
{
    unsigned int parS; __shared__ unsigned int val;
    int j, t = threadIdx.x, b = blockIdx.x;
    int idx = t + Nthreads * b; unsigned int MyParSum = 0;
    for (j = 0; j<b; j+=Nthreads){
        parS = (t+j<b) ? Repd[(t+j)*Nthreads].ParSum.ValInt[1] : 0;
        parS = blockReduceSum<unsigned int>(parS);
        if(t==0) val = parS; __syncthreads(); MyParSum += val; // we sum Roff for all blocks from 0 to (b-1).
    }
    if(idx<R){
        for(j=Nthreads*b;j<idx;j++) MyParSum+=Repd[j].Roff;	// we add Roff for current block threads from 0 to (t-1)
        Repd[idx].ParSum.ValInt[0] = MyParSum;
    }
}

__global__ void resampleKer(Replica* Repd, Replica* Repdnew) // copying replicas (the main part of the resampling process)
{
    int j, jnext, B = blockIdx.x, idx = threadIdx.x + EQthreads * blockIdx.y;
    j = Repd[B].ParSum.ValInt[0]; jnext = j + Repd[B].Roff;
    for(; j < jnext; j++){
        for (int i=0; i<9; ++i){
            Repdnew[j].g[idx + i*Ndiv9]=Repd[B].g[idx + i*Ndiv9];
        }
        //Repdnew[j].gB[idx]=Repd[B].gB[idx];
    }
}

__global__ void CalcAverages(Replica* Repd, int R, double* Averages) // calculation of observables via averaging over the population
{
    int t = threadIdx.x, b = blockIdx.x; int idx = t + Nthreads * b; double currE,currE2,currM,currM2,currM4;
    if(idx<R){ currE = Repd[idx].IE; currM = Repd[idx].M; if(currM<0) currM=-currM;} else{ currE = 0; currM = 0;}
    currE2 = currE*currE; currM2 = currM*currM; currM4 = currM2*currM2;
    currE  = blockReduceSum<double>(currE);	 if(t==0) atomicAdd(&Averages[0], currE);  __syncthreads();
    currE2 = blockReduceSum<double>(currE2); if(t==0) atomicAdd(&Averages[1], currE2); __syncthreads();
    currM  = blockReduceSum<double>(currM);	 if(t==0) atomicAdd(&Averages[2], currM);  __syncthreads();
    currM2 = blockReduceSum<double>(currM2); if(t==0) atomicAdd(&Averages[3], currM2); __syncthreads();
    currM4 = blockReduceSum<double>(currM4); if(t==0) atomicAdd(&Averages[4], currM4);
}

#ifdef MHR

__global__ void UpdateShistE(Replica* Repd, int R, int* ShistE) // adding energy histogram of the current temperature step for the MHR analysis
{
	int t = threadIdx.x, b = blockIdx.x; int idx = t + Nthreads * b;
	if(idx<R){
		atomicAdd(&ShistE[(2*N+Repd[idx].IE)/4],1);
	}
}

#endif

#ifdef AdaptiveStep

__global__ void HistogramOverlap(Replica* Repd, int Rinit, int R, double lnQ, double dB, double* overlap) // calculating histogram overlap
{
	double PartialOverlap;
	int t = threadIdx.x, idx = threadIdx.x + Nthreads * blockIdx.x;
	PartialOverlap = (idx < R) ? min(1.0,((double)Rinit)/R*exp(-dB*(double)Repd[idx].IE - lnQ)) : 0 ;
	PartialOverlap = blockReduceSum<double>(PartialOverlap);
	if(t==0) atomicAdd(overlap,PartialOverlap);
}

double CalcOverlap(Replica* Rep_d, double dB, int R, double Emean){	// Calculates histogram overlap
		double q, lnQ, ioverlaph;
		int NblocksR = (int)ceil(R/(double)Nthreads);
		QKer <<< NblocksR, Nthreads >>> (Rep_d, R, dB, Emean, 0, Qd);
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		QKer <<< 1, Nthreads >>> (Rep_d, R, dB, Emean, 1, Qd);
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		CUDAErrChk( cudaMemcpy(&q,Qd,sizeof(double),cudaMemcpyDeviceToHost) );
		lnQ = -dB * Emean + log(q) - log((double)R);
		CUDAErrChk( cudaMemset(ioverlapd, 0, sizeof(double)) );
		HistogramOverlap<<<NblocksR,Nthreads>>>(Rep_d, Rinit, R, lnQ, dB, ioverlapd);
		CUDAErrChk( cudaPeekAtLastError() );
		CUDAErrChk( cudaDeviceSynchronize() );
		CUDAErrChk( cudaMemcpy(&ioverlaph,ioverlapd,sizeof(double),cudaMemcpyDeviceToHost) );
		return (double)ioverlaph/R;
}

#endif

char *optarg; int opterr = 1, optind = 1, optopt, optreset;

int getopt(int nargc, char * const nargv[], const char *ostr)
{
    static char *place = (char*)""; const char *oli;
    if (optreset || !*place) {
        optreset = 0;
        if (optind >= nargc || *(place = nargv[optind]) != '-') { place = (char*)""; return (-1); }
        if (place[1] && *++place == '-') { ++optind; place = (char*)""; return (-1); }
    }
    if ((optopt = (int)*place++) == (int)':' || !(oli = strchr(ostr, optopt))) {
        if (optopt == (int)'-') return (-1);
        if (!*place) ++optind;
        if (opterr && *ostr != ':') (void)printf("illegal option -- %c\n", optopt);
        return ((int)'?');
    }
    if (*++oli != ':') { optarg = NULL; if (!*place) ++optind; }
    else {
        if (*place) optarg = place; else if (nargc <= ++optind) {
            place = (char*)""; if (*ostr == ':') return ((int)':');
            if (opterr) (void)printf("option requires an argument -- %c\n", optopt);
            return ((int)'?');
        }
        else optarg = nargv[optind];
        place = (char*)"";  ++optind;
    }
    return (optopt);
}

void PrintParameterUsage(){
    cout << " Usage: PAisingSSC [options]\n"
         << " Note: all of the options are optional. Default parameter values are listed in the head of the source code. \n"
         << " Possible command line options are:\n\n"
         << " -R Rinit           ( Rinit           = initial size of population of replicas )\n"
         << " -t EQsweeps        ( EQsweeps        = number of equilibration sweeps )\n"
         << " -d dBinit          ( dBinit          = inverse temperature step )\n"
         << " -f Bfin            ( Bfin            = final value of inverse temperature )\n"
         << " -M runs            ( runs            = number of population annealing algorithm independent runs )\n"
         << " -s RNGseed         ( RNGseed         = seed for random number generation )\n"
         << " -P OutputPrecision ( OutputPrecision = precision (number of digits) of the output )\n"
         << " -o dataDirectory   ( dataDirectory   = data directory name )\n";
}

int main(int argc, char** argv)
{
    // data directory name + create
    char dataDir[200]; int rng_seed = RNGseed; int optdir = 0;

    int optc, opti; double optf;
    while ((optc = getopt (argc, argv, "R:t:d:f:M:s:P:o:?")) != -1)	// Processing optional command line options
        switch (optc)
        {
            case 'R': opti = atoi(optarg); if(opti) Rinit = opti; break;           		// -R Rinit
            case 't': opti = atoi(optarg); EQsweeps = opti; break;                  	// -t EQsweeps
            case 'd': optf = atof(optarg); if(optf > 0.0) dBinit = optf; break;     	// -d dBinit
            case 'f': optf = atof(optarg); if(optf > 0.0) Bfin = optf; break;       	// -f Bfin
            case 'M': opti = atoi(optarg); if(opti) runs = opti; break;             	// -M runs
            case 's': opti = atoi(optarg); if(opti) rng_seed = opti; break;         	// -s RNGseed
            case 'P': opti = atoi(optarg); if(opti) OutputPrecision = opti; break;   	// -P OutputPrecision
            case 'o': if(optarg[strlen(optarg)-1]=='/') sprintf(dataDir,"%s",optarg);	// -o dataDir
                else sprintf(dataDir,"%s/",optarg); optdir = 1; break;
            case '?': PrintParameterUsage();  return 1;
        }
    if(optind < argc){
        for (opti = optind; opti < argc; opti++) fprintf(stderr,"Non-option argument %s\n", argv[opti]);
        return 1;
    }

#ifdef AdaptiveStep
    if(!optdir) sprintf(dataDir, "./dataSSC_L%d_R%d_EqSw%d/", L, Rinit, EQsweeps);
#else
    if(!optdir) sprintf(dataDir, "./dataSSC_L%d_R%d_EqSw%d_dB%f/", L, Rinit, EQsweeps, dBinit);
#endif

#ifdef _WIN32
    _mkdir(dataDir);
#else
    mkdir(dataDir, 0777);
#endif

    int rmin=0, rmax=runs-1; unsigned long long initial_sequence = 0;

    double B[nBmax], Binc[nBmax]; B[0]=Binc[0]=Binit; long long totPop=0;

    cout <<"RNG initial seed: "<< rng_seed<<"\n";

#ifdef LatticePopStore
    int LatticePopStoreTable[nBmax];
	        for(int j=0;j<nBmax;j++) LatticePopStoreTable[j]=1;
#endif

    // creating data arrays for thermodynamic variables and errors
    double E[nBmax]; double M[nBmax]; double M2[nBmax]; double M4[nBmax];
    double C[nBmax];
    double lnQ[nBmax]; 		// partition function ratio
    double BF[nBmax]; 			// dimensionless free energy estimate
    double S[nBmax]; 			// entropy
    BF[0] = - N*log(2.0);		// its value at infinite temperature
    int R[nBmax];				// population size

    // CUDAErrChk( cudaSetDevice(0) );  // uncomment to explicitly select device number in a setup with multiple cards
    CUDAErrChk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)); // prefer larger L1 cache and smaller shared memory
    // GPU execution time
    cudaEvent_t start, stop; float Etime;
    CUDAErrChk( cudaEventCreate(&start) );
    CUDAErrChk( cudaEventCreate(&stop) );
    // start evaluation time measurement
    cudaEventRecord(start, 0);

    double *Averages; double Averages_h[5]; unsigned int* Ridev;
    CUDAErrChk( cudaMalloc((void**)&Averages,5*sizeof(double)) );
    CUDAErrChk( cudaMalloc((void**)&Qd,sizeof(double)) );
    CUDAErrChk( cudaMalloc((void**)&Ridev,sizeof(int)) );
    CUDAErrChk( cudaMalloc((void**)&ioverlapd,sizeof(double)) );

    R[0] = Rinit;
    cout << "Memory use of one replica: " << sizeof(Replica) / 1024.0 << " kB \n";
    cout << "Memory use of the entire population of " << R[0] << " replicas: "
         << R[0]*sizeof(Replica) / 1024.0 / 1024.0 << " MB \n"; fflush(stdout);

    // creating energy spectrum for multi-histogram reweighting
#ifdef MHR
    int Ei[N+1];
		for (int i = 0; i < N+1; ++i){
			Ei[i] = 4*i - 2*N;
		}
#endif

    // host and device pointer to data of replicas
    Replica* Rep_d;
    Replica* RepNew_d;
    Replica* RepDel;



    double boltzGPU[boltzTableL]; // Boltzman factor table - host version
    double* boltztext;

    CUDAErrChk( cudaMalloc((void**)&sublat,2*9*Ndiv9*sizeof(int)) );//new sublat

    int ix,iy;
    int sublatCPU[2*9*Ndiv9];//массив с координатами спинов

    for (int i=0;i<9;++i){//заполнение i-ой подрешетки
        ix = i%3;
        iy = i/3;
        for (int j=0; j<Ndiv9;++j){
            sublatCPU[2*(j + i*Ndiv9)]=ix;
            sublatCPU[1 + 2*(j + i*Ndiv9)]=iy;
            ix = (ix+3)%L;
            if ((j+1)%Ldiv3 == 0) iy+=3;
        }
    }

    // copying sublatCPU to sublat
    CUDAErrChk( cudaMemcpy(sublat, sublatCPU, 2*9*Ndiv9 * sizeof(int),cudaMemcpyHostToDevice) );//new sublat
    //CUDAErrChk( cudaMemcpy(sublatCPU, sublat, 2*9*Ndiv9 * sizeof(int),cudaMemcpyHostToDevice) );//new sublat
    // memory allocation for Boltzmann factor table
    CUDAErrChk( cudaMalloc((void **)&boltztext, boltzTableL * sizeof(double)) );
    // binding references (global & texture memory buffers)
    CUDAErrChk( cudaBindTexture(NULL,boltzT,boltztext,boltzTableL * sizeof(double)) );

    int Ethreads = 1; while(Ethreads < EQthreads) Ethreads <<= 1;



    for (int r = rmin; r <= rmax; ++r){
        double sumlnQ = 0.0; double q; double Emean = 0.0;
        CUDAErrChk( cudaMalloc((void **)&Rep_d,R[0]*sizeof(Replica)) );
        initial_sequence=0;
        ReplicaInit <<< R[0], EQthreads >>> (Rep_d,rng_seed,initial_sequence); initial_sequence+=R[0]*EQthreads;
        //if (r==1) ++rng_seed;


        CUDAErrChk( cudaPeekAtLastError() );
        CUDAErrChk( cudaDeviceSynchronize() );

        // compute energy of all replicas at zero temperature (for 1st resampling)
        energyKer <<< R[0], Ethreads >>> (Rep_d);

        CUDAErrChk( cudaPeekAtLastError() );
        CUDAErrChk( cudaDeviceSynchronize() );

        // array for summing the energy histograms over inverse temperatures
#ifdef MHR
        int ShistE[N+1]; int* ShistEd;
			CUDAErrChk( cudaMalloc((void**)&ShistEd,(N+1)*sizeof(int)) );
			CUDAErrChk( cudaMemset(ShistEd,0,(N+1)*sizeof(int)) );
			UpdateShistE<<<(int)ceil(R[0]/(float)Nthreads),Nthreads>>> (Rep_d, R[0], ShistEd);
			CUDAErrChk( cudaPeekAtLastError() );
			CUDAErrChk( cudaDeviceSynchronize() );
#endif

        // ------------------------------------------------------------------
        // population annealing
        // ------------------------------------------------------------------
        int i=1,iprev=0,nB; double deltaBeta=dBinit; B[i]=Binc[i]=B[iprev]+deltaBeta;
        //double j1=1, j2=0.3, j3=0.1;
        int i123;
        while(B[i]<=Bfin) {
            // Boltzmann factor tabulation

            for (int idxT0 = 0; idxT0 <= 1; ++idxT0){   // idxT0 = (1 + Si)/2
                for (int idxT1 = 0; idxT1 <= 6; ++idxT1 ){  // idxT1 = (4 + sum(Sj))/2 ; Sj are NN with J1
                    for (int idxT2 = 0; idxT2 <= 6; ++idxT2 ){
                        for (int idxT3 = 0; idxT3 <= 6; ++idxT3 ) {
                            i123 = idxT1*49 + 7*idxT2 + idxT3;
                            boltzGPU[idxT0 + 2*i123] =\
                            min(1.0, exp(2*(2*idxT0-1)*(J1*( 2*idxT1-6)+J2*(2*idxT2-6)+J3*(2*idxT3-6)) * B[i] ) );
                        }
                    }
                }
            }
            // copying table to texture memory - boltztext is bounded with boltzT
            CUDAErrChk( cudaMemcpy(boltztext, boltzGPU, boltzTableL * sizeof(double),cudaMemcpyHostToDevice) );
            //CUDAErrChk( cudaMemcpy(boltzGPU, boltztext, boltzTableL * sizeof(double),cudaMemcpyHostToDevice) );

            // compute the partition function ratio - Q

            int NblocksR = (int)ceil(R[i-1]/(double)Nthreads);
            dim3 DimGridRes(R[i-1],N/2/EQthreads,1);
            QKer <<< NblocksR, Nthreads >>> (Rep_d, R[i-1], B[i] - B[i-1], Emean, 0, Qd);
            CUDAErrChk( cudaPeekAtLastError() );
            CUDAErrChk( cudaDeviceSynchronize() );
            QKer <<< 1, Nthreads >>> (Rep_d, R[i-1], B[i] - B[i-1], Emean, 1, Qd);
            CUDAErrChk( cudaPeekAtLastError() );
            CUDAErrChk( cudaDeviceSynchronize() );
            CUDAErrChk( cudaMemcpy(&q,Qd,sizeof(double),cudaMemcpyDeviceToHost) );

            lnQ[i] = -(B[i] - B[i-1])*Emean + log(q) - log((double)R[i-1]);

            CUDAErrChk( cudaMemset(Ridev, 0, sizeof(unsigned int)) );

            CalcTauKer <<< NblocksR, Nthreads >>> (Rep_d, Rinit, R[i-1], lnQ[i], B[i] - B[i-1], Ridev, rng_seed, initial_sequence); initial_sequence+=R[i-1];
            CUDAErrChk( cudaPeekAtLastError() );
            CUDAErrChk( cudaDeviceSynchronize() );
            CUDAErrChk( cudaMemcpy(&R[i], Ridev, sizeof(int),cudaMemcpyDeviceToHost) );
            CalcParSum<<< NblocksR, Nthreads >>> (Rep_d, R[i-1]);
            CUDAErrChk( cudaPeekAtLastError() );

            CUDAErrChk( cudaMalloc((void **)&RepNew_d,R[i]*sizeof(Replica)) );


            resampleKer<<< DimGridRes, EQthreads >>> (Rep_d, RepNew_d);
            CUDAErrChk( cudaPeekAtLastError() );
            CUDAErrChk( cudaDeviceSynchronize() );

            RepDel = Rep_d; Rep_d = RepNew_d; CUDAErrChk( cudaFree(RepDel) );
            Replica* nikolay = new Replica[1];
            CUDAErrChk( cudaMemcpy(nikolay, Rep_d, sizeof(Replica),cudaMemcpyDeviceToHost) );
            //cout<<"rep_d before equilibration\n";
            //print_replica(nikolay[0], r);

            // equilibrate replicas for certain number of sweeps
            checkKerALL <<< R[i], EQthreads >>> (Rep_d,EQsweeps,rng_seed,initial_sequence, sublat); initial_sequence+=R[i]*EQthreads;
            energyKer <<< R[i], Ethreads >>> (Rep_d);
//             cout<<"rep_d after 100 sweeps\n";
                CUDAErrChk( cudaMemcpy(nikolay, Rep_d, sizeof(Replica),cudaMemcpyDeviceToHost) );

              if (B[i] >= 9.9)print_replica(nikolay[0], r);
//             CUDAErrChk( cudaMemcpy(nikolay, Rep_d, sizeof(Replica),cudaMemcpyDeviceToHost) );
//
//
//             for (int j = 0; j<10; ++j){
//             checkKerALL <<< R[i], EQthreads >>> (Rep_d,1,rng_seed,initial_sequence, sublat); initial_sequence+=R[i]*EQthreads;
//             energyKer <<< R[i], Ethreads >>> (Rep_d);
//             CUDAErrChk( cudaMemcpy(nikolay, Rep_d, sizeof(Replica),cudaMemcpyDeviceToHost) );
//             cout<<"rep_d after one more sweep\n";
//             print_replica(nikolay[0], r);
//             }
            CUDAErrChk( cudaPeekAtLastError() );
            CUDAErrChk( cudaDeviceSynchronize() );

            // compute observables (E,M,O,F)
            // compute energy of all replicas


            energyKer <<< R[i], Ethreads >>> (Rep_d);

//             CUDAErrChk( cudaMemcpy(nikolay, Rep_d, sizeof(Replica),cudaMemcpyDeviceToHost) );
//             cout<<"rep_d after equilibration\n";
//             print_replica(nikolay[0], r);


            CUDAErrChk( cudaPeekAtLastError() );
            CUDAErrChk( cudaDeviceSynchronize() );

            // saving results - energies
#ifdef EnergiesPopStore
            Replica* Rep_h = (Replica*)malloc(R[i]*sizeof(Replica));
				CUDAErrChk( cudaMemcpy(Rep_h, Rep_d, R[i]*sizeof(Replica),cudaMemcpyDeviceToHost) );
				ofstream results;
				char str[100];
				char str2[100];
				strcpy(str, dataDir);
				sprintf(str2,"PA_energies_%d.dat",i);
				strcat(str,str2);
				results.open(str);
				results.precision(OutputPrecision);
				for (int j = 0; j < R[i]; ++j) {
					results << Rep_h[j].IE << " ";
				}
				results.close(); free(Rep_h);
#endif

#ifdef LatticePopStore
            if (LatticePopStoreTable[i]&&(r==0)){
					Replica* Rep_h = (Replica*)malloc(R[i]*sizeof(Replica));
					CUDAErrChk( cudaMemcpy(Rep_h, Rep_d, R[i]*sizeof(Replica),cudaMemcpyDeviceToHost) );
					ofstream results;
					char str[100];
					char str2[100];
					strcpy(str, dataDir);
					sprintf(str2,"PA_lattice_%d.dat",i);
					strcat(str,str2);
					results.open(str,std::ofstream::out | std::ofstream::app);
					results.precision(OutputPrecision);
					int j,k,ABflag=0; results<<"i="<<i<<"; beta="<<B[i]<<"; R="<<R[i]<<";"<<endl;
					for (j = 0; j < R[i]; ++j) {
						results<<"P={";
	                                        k=0; while(k<(N/2)){
							if((k%(L/2))==0) {results << "{"; ABflag=1-ABflag;}
							results << (int)Rep_h[j].g[k];//if(ABflag) results << (int)Rep_h[j].gA[k] << "," << (int)Rep_h[j].gB[k];
							//else results << (int)Rep_h[j].gB[k] << "," << (int)Rep_h[j].gA[k];
							k++; if((k%(L/2))==0) {results<<"}"; if(k<(N/2)) results<<",";}
							else results<<",";
						}
						results <<"};"<<endl;
					}
					results<<endl; results.close(); free(Rep_h);
				}
#endif

#ifdef MHR
            UpdateShistE<<<(int)ceil(R[i]/(float)Nthreads),Nthreads>>>(Rep_d,R[i],ShistEd);
				CUDAErrChk( cudaPeekAtLastError() );
				CUDAErrChk( cudaDeviceSynchronize() );
#endif

            CUDAErrChk( cudaMemset(Averages, 0, 5*sizeof(double)) );
            CalcAverages<<<(int)ceil(R[i]/(float)Nthreads),Nthreads>>>(Rep_d,R[i],Averages);
            CUDAErrChk( cudaPeekAtLastError() );
            CUDAErrChk( cudaDeviceSynchronize() );
            CUDAErrChk( cudaMemcpy(Averages_h,Averages,5*sizeof(double),cudaMemcpyDeviceToHost) );

            E[i]  = Emean = Averages_h[0] / R[i];
            C[i]  = (Averages_h[1] / R[i] - E[i]*E[i]) * B[i] * B[i];
            M[i]  = Averages_h[2] / R[i];
            M2[i] = Averages_h[3] / R[i];
            M4[i] = Averages_h[4] / R[i];

            // dimensionless free energy
            sumlnQ -= lnQ[i];
            BF[i] = - N*log(2.0) + sumlnQ;
            // entropy
            S[i] = B[i]*E[i] - BF[i];

            iprev=i; totPop+=R[i]; i++;

            if(i>=nBmax){
#ifdef AdaptiveStep
                fprintf(stderr,"Error: number of temperature steps exceeds nBmax=%d.\n Please consider increasing the population size or decreasing the value of MinOverlap or increasing the value of nBmax.\n",nBmax);
#else
                fprintf(stderr,"Error: number of temperature steps exceeds nBmax=%d.\n Please consider increasing the inverse temperature step or increasing the value of nBmax.\n",nBmax);
#endif
                return 1;
            }

            // determining next inverse temperature step
            if(r==rmin){
#ifdef AdaptiveStep
                double overlap, dBmin = 0, dBmax = deltaBeta, dBmean;
					while(1){
						overlap = CalcOverlap ( Rep_d, dBmax, R[iprev] , Emean);
						if ( (overlap >= MaxOverlap) && (B[iprev] + dBmax < Bfin) ) dBmax *= 1.1; else break;
					}
					if ( overlap >= MinOverlap ) dBmean = dBmax;
					else while(1){	// obtaining optimal inverse temperature step with the bisection method
						dBmean = 0.5 * (dBmin + dBmax);
						overlap = CalcOverlap ( Rep_d, dBmean, R[iprev] , Emean);
						if ( overlap < MinOverlap ) dBmax = dBmean;
						else if ( overlap >= MaxOverlap ) dBmin = dBmean;
						else break;
					}
					if( (B[iprev] < Bfin) && (B[iprev] + dBmean > Bfin) ) deltaBeta = Bfin - B[iprev]; else deltaBeta = dBmean;
#endif
                B[i]=Binc[i]=B[iprev]+deltaBeta;
            } else B[i]=Binc[i];

        }

        CUDAErrChk( cudaFree(Rep_d) );
        nB=i;

        // saving results
        {
            ofstream results;
            char str[100];
            char str2[100];
            strcpy(str, dataDir);
            sprintf(str2, "PA_results_run_%d.dat", r);
            strcat(str,str2);
            results.open(str);
            results.precision(OutputPrecision);
            for (int i = 0; i < nB; ++i) {
                results << B[i] << " "
                        << E[i] / N << " "
                        << C[i] / N << " "
                        << M[i] / N << " "
                        << M2[i] / N / N << " "
                        << M4[i] / N / N / N / N << " "
                        << BF[i] / N << " "
                        << S[i] / N << " "
                        << R[i] << " "
                        << lnQ[i] << "\n";
            }
            results.close();
        }

        // multi-histogam reweighting (MHR) analysis
#ifdef MHR
        // declaring arrays used in MHR analysis
			double lnOmega[N+1];
			double E_MHR[nB*MHR_Niter];
			double C_MHR[nB*MHR_Niter];
			double BF_MHR[nB*MHR_Niter];
			bool relTerm[N+1];

			CUDAErrChk( cudaMemcpy(ShistE,ShistEd,(N+1)*sizeof(int),cudaMemcpyDeviceToHost) );

			for (int l = 0; l < MHR_Niter; ++l){
				// calculate lnOmega
				double Sigma[nB];
				double mSigma;
				for (int k = 0; k < N+1; ++k){
					// maxima of -S = BF - B*E
					Sigma[0] = BF[0]-B[0]*Ei[k];
					mSigma = Sigma[0];
					for (int i = 1; i < nB; ++i){
						Sigma[i] = BF[i]-B[i]*Ei[k];
						if (mSigma < Sigma[i]){
							mSigma = Sigma[i];
						}
					}
					double sD = 0;
					for (int i = 0; i < nB; ++i){
						sD += R[i]*exp(Sigma[i]-mSigma);
					}
					if ((ShistE[k] == 0) || (sD == 0)){
						relTerm[k] = false;
						lnOmega[k] = 0;
					} else {
						relTerm[k] = true;
						lnOmega[k] = log(ShistE[k]) - mSigma - log(sD);
					}
				}
				// reweigting of observables
				double expOm[N+1];
				double Om[N+1];
				double mOm;
				for (int i = 0; i < nB; ++i){
					// determine the maxima of the reweighting exponent
					mOm = lnOmega[0] - B[i]*Ei[0];
					for (int k = 0; k < N+1; ++k){
						Om[k] = lnOmega[k] - B[i]*Ei[k];
						if (mOm < Om[k]){
							mOm = Om[k];
						}
					}
					// calculate reweighting exponentials
					double p = 0;
					for (int k = 0; k < N+1; ++k){
						expOm[k] = exp(Om[k] - mOm);
						if (relTerm[k])
							p += expOm[k];
					}
					double s = 0;
					for (int k = 0; k < N+1; ++k){
						if (relTerm[k])
							s += Ei[k]*expOm[k];
					}
					E_MHR[i+l*nB] = s / p / N;
					BF_MHR[i+l*nB] = - mOm - log(p);
					BF[i] = BF_MHR[i+l*nB];
					s = 0;
					for (int k = 0; k < N+1; ++k){
						if (relTerm[k])
							s += pow(Ei[k]-E_MHR[i+l*nB]*N,2)*expOm[k];
					}
					C_MHR[i+l*nB] = B[i]*B[i] * s / p / N;
				}
			}
			// saving results
			{
				ofstream results;
				char MHRDataFile[100];
				char str2[100];
				strcpy(MHRDataFile, dataDir);
				sprintf(str2,"PA_MHR_results_run_%d.dat",r);
				strcat(MHRDataFile,str2);
				results.open(MHRDataFile);
				results.precision(OutputPrecision);
				for (int i = 0; i < nB; ++i){
					results << B[i] << " ";
					for (int l = 0; l < MHR_Niter; ++l){
						results << E_MHR[i+l*nB] << " ";
						results << C_MHR[i+l*nB] << " ";
						results << BF_MHR[i+l*nB] / N << " ";
					}
					results << "\n";
				}
				results.close();
			}

			CUDAErrChk( cudaFree(ShistEd) );

#endif
    }
    CUDAErrChk( cudaFree(Averages) );
    CUDAErrChk( cudaFree(Ridev) );
    CUDAErrChk( cudaFree(Qd) );
    CUDAErrChk( cudaFree(sublat) );//new sublat
    CUDAErrChk( cudaFree(ioverlapd) );
    CUDAErrChk( cudaUnbindTexture(boltzT) );
    CUDAErrChk( cudaFree(boltztext));

    CUDAErrChk( cudaDeviceSynchronize() );
    CUDAErrChk( cudaEventRecord(stop, 0) );
    CUDAErrChk( cudaEventSynchronize(stop) );
    CUDAErrChk( cudaEventElapsedTime(&Etime, start, stop) );

    cout << "Elapsed time: " << setprecision(8) << Etime/1000 << " s\n";
    cout << "Time per spin-flip: " << setprecision(8) << Etime*1e6/EQsweeps/N/totPop << " ns\n";

    CUDAErrChk( cudaEventDestroy(start) );
    CUDAErrChk( cudaEventDestroy(stop) );

    return 0;
}
