#include "FMatrixGPU.h"

struct _device_FM_PX_Vec_params {
   const double *ent; //!< FMatrixArray device entries
   const double *b;   //!< Array device entries
   double *c;         //!< Array device entries
   int cols;          //!< Number of columns
   int mats;          //!< Number of matrices
};


/** FMatrix Vector product - Kernel function
 * \param p       structured containing all the parameters used in the function
 *
 * \post          p.c contains the result of A*b
 */
__global__ void __device_FM_PX_Vec(_device_FM_PX_Vec_params p) {

   const int vsz      = p.cols;
   const int msz      = vsz*vsz;
   const int blkStart = msz*_N*blockIdx.x + (threadIdx.x/_L)*msz*_L + (threadIdx.x%_L);
   const int vStart   = vsz*_N*blockIdx.x + (threadIdx.x/_L)*vsz*_L + (threadIdx.x%_L);
   int maxsize = _N*blockIdx.x + threadIdx.x;

   if(maxsize < p.mats) { 
      int idx = blkStart; 
      for(int i = 0; i < vsz; i++) {
         double s = 0.0;
         for(int j = 0; j < vsz; j++, idx+=_L)
            s += p.ent[idx] * p.b[vStart+j*_L];		
         p.c[vStart+i*_L] = s;
      }  
   }
}

/** FMatrix Vector product - Interface function for kernel calls
 * \param dEnt       pointer to previously filled device FMatrixArray entries
 * \param NCols      number of columns of each FMatrix
 * \param NRows      number of rows of each FMatrix - not used: assume NCols = NRows
 * \param numMats    number of matrices
 * \param b          pointer to previously filled device array entries
 * \param c          pointer to previously allocated device array entries
 *
 * \post             c contains the result of A*b
 */
extern "C" {
void _device_FM_PX_Vec(Real *dEnt, const int NCols, const int NRows, const int numMats, Real *b, Real *c) {

   _device_FM_PX_Vec_params p;

   p.ent  = dEnt;
   p.b    = b;
   p.c    = c;
   p.cols = NCols;
   p.mats = numMats;

   __device_FM_PX_Vec<<< (p.mats+_N-1)/_N, _N >>>(p);
}
}


struct _device_FM_Plus_FM_params {
   double *A;   //!< FMatrixArray device entries
   double *B;   //!< FMatrixArray device entries
   double *C;   //!< FMatrixArray device entries
   int cols;    //!< Number of columns
   int mats;    //!< Number of matrices
};

/** FMatrix Addition - Kernel function
 * \param p       structured containing all the parameters used in the function
 *
 * \post          p.C contains the result of A+B
 */
__global__ void __device_FM_Plus_FM(_device_FM_Plus_FM_params p){

   const int msz      = p.cols*p.cols;
   const int blkStart = msz*_N*blockIdx.x + (threadIdx.x/_L)*msz*_L + (threadIdx.x%_L);
   const int blkEnd   = blkStart + msz*_L;
   int maxsize = _N*blockIdx.x + threadIdx.x;

   if(maxsize < p.mats) { 
      for(int i = blkStart; i < blkEnd; i+=_L)
         p.C[i] = p.A[i] + p.B[i];
   }
}

/** FMatrix Addition - Interface function for kernel calls
 * \param A          pointer to previously filled device FMatrixArray entries
 * \param B          pointer to previously filled device FMatrixArray entries
 * \param C          pointer to previously allocated device FMatrixArray entries
 * \param NCols      number of columns of each FMatrix
 * \param NRows      number of rows of each FMatrix - not used: assume NCols = NRows
 * \param numMats    number of matrices
 *
 * \post             C contains the result of A+B
 */
extern "C" {
void  _device_FM_Plus_FM(Real *A, Real *B, Real *C, const int NCols, const int NRows, const int numMats){

   _device_FM_Plus_FM_params p;
   p.A = A;
   p.B = B;
   p.C = C;
   p.cols = NCols;
   p.mats = numMats;
   
   __device_FM_Plus_FM<<< (p.mats+_N-1)/_N,_N >>>(p);
}
}

struct _device_FM_Det_params {
   double *ent;   //!< FMatrixArray device entries
   double *det;   //!< device array of determinants
   int cols;      //!< Number of columns
   int mats;      //!< Number of matrices
};

/** 2x2 FMatrix determinant - Kernel function
 * \param p       structured containing all the parameters used in the function
 *
 * \post          p.det contains the determinant of each 2x2 FMatrix
 */
__global__ void __device_FM_Det_2x2(_device_FM_Det_params p) {

   const int msz      = p.cols*p.cols;
   const int blkStart = msz*_N*blockIdx.x + (threadIdx.x/_L)*msz*_L + (threadIdx.x%_L);
   const int detIdx   = _N*blockIdx.x + threadIdx.x;

   if(detIdx < p.mats) { 
      const double a11 = p.ent[blkStart], a12 = p.ent[blkStart+_L], a21 = p.ent[blkStart+2*_L], a22 = p.ent[blkStart+3*_L];

      p.det[detIdx] = a11*a22 - a12*a21;
   }
}

/** 3x3 FMatrix determinant - Kernel function
 * \param p       structured containing all the parameters used in the function
 *
 * \post          p.det contains the determinant of each 3x3 FMatrix
 */
__global__ void __device_FM_Det_3x3(_device_FM_Det_params p) {

   const int msz      = p.cols*p.cols;
   const int blkStart = msz*_N*blockIdx.x + (threadIdx.x/_L)*msz*_L + (threadIdx.x%_L);
   const int detIdx   = _N*blockIdx.x + threadIdx.x;

   if(detIdx < p.mats) { 
      const double a11 = p.ent[blkStart],      a12 = p.ent[blkStart+_L],   a13 = p.ent[blkStart+2*_L], 
	                a21 = p.ent[blkStart+3*_L], a22 = p.ent[blkStart+4*_L], a23 = p.ent[blkStart+5*_L],
                   a31 = p.ent[blkStart+6*_L], a32 = p.ent[blkStart+7*_L], a33 = p.ent[blkStart+8*_L];

      p.det[detIdx] = (a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33;
   }
}

/** 4x4 FMatrix determinant - Kernel function
 * \param p       structured containing all the parameters used in the function
 *
 * \post          p.det contains the determinant of each 4x4 FMatrix
 */
__global__ void __device_FM_Det_4x4(_device_FM_Det_params p) {

   const int msz      = p.cols*p.cols;
   const int blkStart = msz*_N*blockIdx.x + (threadIdx.x/_L)*msz*_L + (threadIdx.x%_L);
   const int detIdx   = _N*blockIdx.x + threadIdx.x;

   if(detIdx < p.mats) { 
      const double a11 = p.ent[blkStart],       a12 = p.ent[blkStart+_L],    a13 = p.ent[blkStart+2*_L], 
                   a14 = p.ent[blkStart+3*_L],  a21 = p.ent[blkStart+4*_L],  a22 = p.ent[blkStart+5*_L], 
                   a23 = p.ent[blkStart+6*_L],  a24 = p.ent[blkStart+7*_L],  a31 = p.ent[blkStart+8*_L], 
                   a32 = p.ent[blkStart+9*_L],  a33 = p.ent[blkStart+10*_L], a34 = p.ent[blkStart+11*_L],
                   a41 = p.ent[blkStart+12*_L], a42 = p.ent[blkStart+13*_L], a43 = p.ent[blkStart+14*_L], 
                   a44 = p.ent[blkStart+15*_L];

      p.det[detIdx] =  ((a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33)*a44 
                      -((a12*a24 - a14*a22)*a31 - (a11*a24 - a14*a21)*a32 + (a11*a22 - a12*a21)*a34)*a43
                      -((a13*a24 - a14*a23)*a32 - (a12*a24 - a14*a22)*a33 + (a12*a23 - a13*a22)*a34)*a41
                      +((a13*a24 - a14*a23)*a31 - (a11*a24 - a14*a21)*a33 + (a11*a23 - a13*a21)*a34)*a42;
   }
}

/** FMatrix Determinant - Interface function for kernel calls
 * \param dEnt       pointer to previously filled device FMatrixArray entries
 * \param NCols      number of columns of each FMatrix
 * \param NRows      number of rows of each FMatrix - not used: assume NCols = NRows
 * \param numMats    number of matrices
 * \param det        pointer to previouly allocated array of determinants
 *
 * \post             det contains the determinant of each FMatrix
 */
extern "C" {
void _device_FM_Det(Real *dEnt, const int NCols, const int NRows, const int numMats, Real *det) { 

   _device_FM_Det_params p;
  
   p.ent  = dEnt;
   p.det  = det;
   p.cols = NCols;
   p.mats = numMats;


   switch (p.cols) {
      case 2:
         __device_FM_Det_2x2<<< (p.mats+_N-1)/_N,_N >>>(p);
      break;
    	case 3:
         __device_FM_Det_3x3<<< (p.mats+_N-1)/_N,_N >>>(p);
      break;
    	case 4:
         __device_FM_Det_4x4<<< (p.mats+_N-1)/_N,_N >>>(p);
      break;
      default: {
         fprintf(stderr,"\nError in Det routine; M.NRows must = 2,3, or 4...\n");
         exit(1);
      }
   }
}
}

struct _device_FM_Inv_params {
   double *ent;        //!< device FMatrixArray entries
   double *ent_inv;    //!< device FMatrixArray
   int cols;           //!< Number of columns
   int mats;           //!< Number of matrices
};

/** 2x2 FMatrix inverse - Kernel function
 * \param p       structured containing all the parameters used in the function
 *
 * \post          p.ent_inv contains the inverse of each 2x2 FMatrix
 */
__global__ void __device_FM_Inv_2x2(_device_FM_Inv_params p) {

   const int msz = p.cols*p.cols;
   const int blkStart = msz*_N*blockIdx.x + (threadIdx.x/_L)*msz*_L +(threadIdx.x%_L);
   int maxsize = _N*blockIdx.x + threadIdx.x;

   if(maxsize < p.mats) { 
      const double a11 = p.ent[blkStart], a12 = p.ent[blkStart+_L], a21 = p.ent[blkStart+2*_L], a22 = p.ent[blkStart+3*_L];

      double ldet = (a11*a22 - a12*a21);

      if(abs(ldet)  != 0.) {
        ldet = 1./ldet;

        // reordering not needed
        p.ent_inv[blkStart]      =  a22*ldet; 
        p.ent_inv[blkStart+_L]   = -a12*ldet; 
        p.ent_inv[blkStart+2*_L] = -a21*ldet; 
        p.ent_inv[blkStart+3*_L] =  a11*ldet;            
      }
   }
}

/** 3x3 FMatrix inverse - Kernel function
 * \param p       structured containing all the parameters used in the function
 *
 * \post          p.ent_inv contains the inverse of each 3x3 FMatrix
 */
__global__ void __device_FM_Inv_3x3(_device_FM_Inv_params p) {

   const int msz = p.cols*p.cols;
   const int blkStart = msz*_N*blockIdx.x + (threadIdx.x/_L)*msz*_L +(threadIdx.x%_L);
   int maxsize = _N*blockIdx.x + threadIdx.x;

   if(maxsize < p.mats) { 
      const double a11 = p.ent[blkStart],      a12 = p.ent[blkStart+_L],   a13 = p.ent[blkStart+2*_L], 
                   a21 = p.ent[blkStart+3*_L], a22 = p.ent[blkStart+4*_L], a23 = p.ent[blkStart+5*_L],
                   a31 = p.ent[blkStart+6*_L], a32 = p.ent[blkStart+7*_L], a33 = p.ent[blkStart+8*_L];
      
      double ldet = ((a12*a23 - a13*a22)*a31 + (a13*a21 - a11*a23)*a32 + (a11*a22 - a12*a21)*a33);
   
      if(abs(ldet)  != 0.) {
         ldet = 1./ldet;

        // first tree expressions have already been used to calculate ldet
        p.ent_inv[blkStart+2*_L] = (a12*a23 - a13*a22)*ldet;
        p.ent_inv[blkStart+5*_L] = (a13*a21 - a11*a23)*ldet;
        p.ent_inv[blkStart+8*_L] = (a11*a22 - a12*a21)*ldet;
        // order?
        p.ent_inv[blkStart]      = (a22*a33 - a23*a32)*ldet; 
        p.ent_inv[blkStart+_L]   = (a13*a32 - a12*a33)*ldet; 
        p.ent_inv[blkStart+3*_L] = (a23*a31 - a21*a33)*ldet; 
        p.ent_inv[blkStart+4*_L] = (a11*a33 - a13*a31)*ldet; 
        p.ent_inv[blkStart+6*_L] = (a21*a32 - a22*a31)*ldet; 
        p.ent_inv[blkStart+7*_L] = (a12*a31 - a11*a32)*ldet; 
      }
   }
}

/** 4x4 FMatrix inverse - Kernel function
 * \param p       structured containing all the parameters used in the function
 *
 * \post          p.ent_inv contains the inverse of each 4x4 FMatrix
 */
__global__ void __device_FM_Inv_4x4(_device_FM_Inv_params p) {

   const int msz = p.cols*p.cols;
   const int blkStart = msz*_N*blockIdx.x + (threadIdx.x/_L)*msz*_L +(threadIdx.x%_L);
   int maxsize = _N*blockIdx.x + threadIdx.x;

   if(maxsize < p.mats) { 
      const double a11 = p.ent[blkStart],       a12 = p.ent[blkStart+_L],    a13 = p.ent[blkStart+2*_L], 
                   a14 = p.ent[blkStart+3*_L],  a21 = p.ent[blkStart+4*_L],  a22 = p.ent[blkStart+5*_L], 
                   a23 = p.ent[blkStart+6*_L],  a24 = p.ent[blkStart+7*_L],  a31 = p.ent[blkStart+8*_L], 
                   a32 = p.ent[blkStart+9*_L],  a33 = p.ent[blkStart+10*_L], a34 = p.ent[blkStart+11*_L],
                   a41 = p.ent[blkStart+12*_L], a42 = p.ent[blkStart+13*_L], a43 = p.ent[blkStart+14*_L], 
                   a44 = p.ent[blkStart+15*_L];

      double ldet = (((a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33)*a44 
                     -((a12*a24 - a14*a22)*a31 - (a11*a24 - a14*a21)*a32 + (a11*a22 - a12*a21)*a34)*a43
                     -((a13*a24 - a14*a23)*a32 - (a12*a24 - a14*a22)*a33 + (a12*a23 - a13*a22)*a34)*a41
                     +((a13*a24 - a14*a23)*a31 - (a11*a24 - a14*a21)*a33 + (a11*a23 - a13*a21)*a34)*a42);

      if(abs(ldet) != 0.) {
        ldet = 1./ldet;
  
        // GH: In the next 4 calculations the terms besides the ldet have been used in the previous calculation of ldet,
        //     i.e., there seems to be a register reuse now.
        p.ent_inv[blkStart+3*_L]  =-((a13*a24 - a14*a23)*a32 - (a12*a24 - a14*a22)*a33 + (a12*a23 - a13*a22)*a34)*ldet;
        p.ent_inv[blkStart+7*_L]  = ((a13*a24 - a14*a23)*a31 - (a11*a24 - a14*a21)*a33 + (a11*a23 - a13*a21)*a34)*ldet;
        p.ent_inv[blkStart+11*_L] =-((a12*a24 - a14*a22)*a31 - (a11*a24 - a14*a21)*a32 + (a11*a22 - a12*a21)*a34)*ldet;
        p.ent_inv[blkStart+15*_L] = ((a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33)*ldet;

        // GH: Maybe there is no difference how to arrange the rest --> timing needed?
        p.ent_inv[blkStart]       = ((a23*a34 - a24*a33)*a42 - (a22*a34 - a24*a32)*a43 + (a22*a33 - a23*a32)*a44)*ldet;
        p.ent_inv[blkStart+4*_L]  =-((a23*a34 - a24*a33)*a41 - (a21*a34 - a24*a31)*a43 + (a21*a33 - a23*a31)*a44)*ldet;
        p.ent_inv[blkStart+8*_L]  = ((a22*a34 - a24*a32)*a41 - (a21*a34 - a24*a31)*a42 + (a21*a32 - a22*a31)*a44)*ldet;
        p.ent_inv[blkStart+12*_L] =-((a22*a33 - a23*a32)*a41 - (a21*a33 - a23*a31)*a42 + (a21*a32 - a22*a31)*a43)*ldet;

        p.ent_inv[blkStart+_L]    =-((a13*a34 - a14*a33)*a42 - (a12*a34 - a14*a32)*a43 + (a12*a33 - a13*a32)*a44)*ldet;
        p.ent_inv[blkStart+5*_L]  = ((a13*a34 - a14*a33)*a41 - (a11*a34 - a14*a31)*a43 + (a11*a33 - a13*a31)*a44)*ldet;
        p.ent_inv[blkStart+9*_L]  =-((a12*a34 - a14*a32)*a41 - (a11*a34 - a14*a31)*a42 + (a11*a32 - a12*a31)*a44)*ldet;
        p.ent_inv[blkStart+13*_L] = ((a12*a33 - a13*a32)*a41 - (a11*a33 - a13*a31)*a42 + (a11*a32 - a12*a31)*a43)*ldet;

        p.ent_inv[blkStart+2*_L]  = ((a13*a24 - a14*a23)*a42 - (a12*a24 - a14*a22)*a43 + (a12*a23 - a13*a22)*a44)*ldet;
        p.ent_inv[blkStart+6*_L]  =-((a13*a24 - a14*a23)*a41 - (a11*a24 - a14*a21)*a43 + (a11*a23 - a13*a21)*a44)*ldet;
        p.ent_inv[blkStart+10*_L] = ((a12*a24 - a14*a22)*a41 - (a11*a24 - a14*a21)*a42 + (a11*a22 - a12*a21)*a44)*ldet;
        p.ent_inv[blkStart+14*_L] =-((a12*a23 - a13*a22)*a41 - (a11*a23 - a13*a21)*a42 + (a11*a22 - a12*a21)*a43)*ldet;
      }
   }
}

/** FMatrix Inversion - Interface function for kernel calls
 * \param dEnt       pointer to previously filled device FMatrixArray entries
 * \param dEnt_Inv   pointer to previously declared device FMatrixArray entries
 * \param NCols      number of columns of each FMatrix
 * \param NRows      number of rows of each FMatrix - not used: assume NCols = NRows
 * \param numMats    number of matrices
 *
 * \post             dEnt_Inv contains the inverse of dEnt
 */
extern "C" {
void  _device_FM_Inv(Real *dEnt, Real *dEnt_Inv, const int NCols, const int NRows, const int numMats) {

   _device_FM_Inv_params p;

   p.mats    = numMats;
   p.cols    = NCols;
   p.ent     = dEnt;
   p.ent_inv = dEnt_Inv;

   switch (p.cols) {
      case 2:
         __device_FM_Inv_2x2<<< (p.mats+_N-1)/_N,_N >>>(p);
      break;
    	case 3:
         __device_FM_Inv_3x3<<< (p.mats+_N-1)/_N,_N >>>(p);
      break;
    	case 4:
         __device_FM_Inv_4x4<<< (p.mats+_N-1)/_N,_N >>>(p);
      break;
      default: {
         fprintf(stderr,"\nError in  routine; M.NRows must = 2,3, or 4...\n");
         exit(1);
      }
   }
}
}


