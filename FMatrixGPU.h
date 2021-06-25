/** Routines to perform operations on small, local Full Matrices
 *
 *  For larger, possibly distributed dense matrices, set the matrix
 *  structure to DenseSym or DenseASym
 */
#ifndef FMatrixGPU_H
#define FMatrixGPU_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include "../NumComp/NumComp.h"

#ifdef FEMLIB_CUDA
// cuda header files
#include "cuda.h"
#include "cuda_runtime.h"
#endif

// cuda definitions
#define convertToStridedArray 0     //!< flag for convertion function
#define convertToNonStridedArray 1  //!< flag for convertion function
#define _N 64                        //!< maximum number of threads per block
#define _L 32                        //!< stride size

#ifdef FEMLIB_CUDA

#define myAssert(condition) \
  if (!(condition)) { return; }

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
#endif

//typedef double Real; 
//typedef int Integer;

typedef struct {
	int     NCols;        //!< number of cols        
	int     NRows;        //!< number of rows
	int	  numMats;		 //!< number of matrices
	Real   *Ent;          //!< matrix entries on host
	#ifdef FEMLIB_CUDA
	Real   *Ent_str;      //!< strided matrix entries on host
	Real   *dEnt;         //!< strided matrix entries on device
	#endif
} FMatrixArray;


#ifdef __cplusplus
extern "C"
{
#endif

  void  Construct_FMatrixArray(FMatrixArray *M, Integer NR, Integer NC, Integer NumMats);
  void  Construct_FMatrixArrayUsingPTmem(FMatrixArray *M, Integer NR, Integer NC, 
                                                 Integer NM, Real *PTelemBuffer);
  void  Convert_Array(Real *NSent, Real *Sent, Integer rows, Integer cols, Integer OP);
  void  Convert_Integer_Array(Integer *NSent, Integer *Sent, Integer rows, Integer cols, Integer OP);
  void  Destroy_FMatrixArray(FMatrixArray *M);
  void  Zero_FMatrixArray(FMatrixArray *M);
  void  FM_PX_Vec(FMatrixArray *M, Real *b, Real *c);
  void  _device_FM_PX_Vec(Real *dEnt, const int NCols, const int NRows, const int numMats, Real *b, Real *c);
  void  FM_Plus_FMArray(FMatrixArray *A, FMatrixArray *B, FMatrixArray *C);
  void  _device_FM_Plus_FM(Real *A, Real *B, Real *C, const int NCols, const int NRows, const int numMats);
  void  FMatrix_DetArray(FMatrixArray *M, Real *det);
  void  _device_FM_Det(Real *dEnt, const int NCols, const int NRows, const int numMats, Real *det); 
  void  FMatrix_InvArray(FMatrixArray *M, FMatrixArray *M_Inv);
  void  _device_FM_Inv(Real *dEnt, Real *dEnt_Inv, const int NCols, const int NRows, const int numMats); 

#ifdef __cplusplus
}
#endif
#endif
