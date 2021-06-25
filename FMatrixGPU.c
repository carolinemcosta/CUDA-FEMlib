#include "FMatrixGPU.h"

/** Allocate Element Matrix
 *
 * \param M		pointer to previouly declared matrix
 * \param NR   number of rows of an element matrix
 * \param NC   number of cols of an element matrix
 * \param NM   number of matrices
 *
 * \post memory for element matrix allocated
 */
void Construct_FMatrixArray(FMatrixArray *M, Integer NR, Integer NC, Integer NM) 
{
  memset(M, 0, sizeof(FMatrixArray));
  M->NRows   = NR;
  M->NCols   = NC;
  M->numMats = NM;

  M->Ent = calloc(M->numMats*M->NRows*M->NCols, sizeof(Real));

#ifdef FEMLIB_CUDA     
    M->Ent_str = calloc(((int)(M->numMats/_L)+1)*_L*M->NRows*M->NCols, sizeof(Real));
    cudaMalloc((void**)&M->dEnt, ((int)(M->numMats/_L)+1)*_L*M->NRows*M->NCols*sizeof(Real));
    cudaCheckErrors("cudaMalloc fail");
#endif
}


/** Construct FMatrix array the same way as Construct_FMatrixArray, but do not 
 *  allocate memory for the entries. Instead, we only assign a pointer to a 
 *  data buffer which has been previously allocated by PT already.
 *
 * \param M	   pointer to previouly declared matrix
 * \param NR   number of rows of an element matrix
 * \param NC   number of cols of an element matrix
 * \param NM   number of matrices
 * \param K    global accumulated matrix structure
 *
 * \post memory for element matrix allocated
 */
void
Construct_FMatrixArrayUsingPTmem(FMatrixArray *M, Integer NR, Integer NC, 
                                          Integer NM, Real *PTelemBuffer)
{    
  memset(M, 0, sizeof(FMatrixArray));
  M->NRows   = NR;
  M->NCols   = NC;
  M->numMats = NM;

#ifndef FEMLIB_CUDA
  M->Ent = PTelemBuffer;
#else        
  // some adjustments need within PT -> Aurel
  M->Ent_str = NULL;
  M->dEnt    = PTelemBuffer;
#endif
} 


/** Convert generic non-strided array to strided array or 
	 strided array to non-strided array (CUDA only)
 *
 * \param NSent    pointer to non-strided array previously allocated and set on host
 * \param Sent     pointer to strided array previously declared (allocated and set) on host
 * \param NSsize   size of non-strided array
 * \param L        array stride
 * \param OP       operation to be executed: convertToStridedArray or convertToNonStridedArray
 *
 * \post strided memory is allocated for Sent and its values are set or
 *       the values of the strided array are copied to the non-strided array
 */
void Convert_Array(Real *NSent, Real *Sent, Integer rows, Integer cols, Integer OP) {

   const int msz = cols*_L;
	int sidx, nsidx;

   if(OP == 0) { // convertToStridedArray
      for(int j = 0; j < rows; j++) {
         const int blkStart = (int)(j/_L)*msz + (j%_L);
         for(int k = 0, m = 0; k < msz, m < cols; k+=_L, m++) {
		      sidx = k+blkStart;
			   nsidx = m+j*cols;
            Sent[sidx] = NSent[nsidx];
         }
      }
   } 
   else if(OP == 1) {  // convertToNonStridedArray
      for(int j = 0; j < rows; j++) {
         const int blkStart = (j/_L)*msz + (j%_L);
         for(int k = 0, m = 0; k < msz, m < cols; k+=_L, m++) {
		      sidx = k+blkStart;
			   nsidx = m+j*cols;
            NSent[nsidx] = Sent[sidx];
         }
      }
   }
	else {
		fprintf(stderr, "Invalid operation in (%s)", __func__);
		exit(0);
	}
}

void Convert_Integer_Array(Integer *NSent, Integer *Sent, Integer rows, Integer cols, Integer OP) {

   const int msz = cols*_L;
	int sidx, nsidx;

   if(OP == 0) { // convertToStridedArray
      for(int j = 0; j < rows; j++) {
         const int blkStart = (j/_L)*msz + (j%_L);
         for(int k = 0, m = 0; k < msz, m < cols; k+=_L, m++) {
		      sidx = k+blkStart;
			   nsidx = m+j*cols;
            Sent[sidx] = NSent[nsidx];
         }
      }
   } 
   else if(OP == 1) {  // convertToNonStridedArray
      for(int j = 0; j < rows; j++) {
         const int blkStart = (j/_L)*msz + (j%_L);
         for(int k = 0, m = 0; k < msz, m < cols; k+=_L, m++) {
		      sidx = k+blkStart;
			   nsidx = m+j*cols;
            NSent[nsidx] = Sent[sidx];
         }
      }
   }
	else {
		fprintf(stderr, "Invalid operation in (%s)", __func__);
		exit(0);
	}
}

/** Deallocate Element Matrix
 *
 * \param M    pointer to initialized FMatrixArray
 *
 * \post       memory for element matrices is deallocated
 */
void Destroy_FMatrixArray(FMatrixArray *M) {

	free(M->Ent);
	memset(M, 0, sizeof(FMatrixArray));

	#ifdef FEMLIB_CUDA
	free(M->Ent_str);
	cudaFree(M->dEnt);
	#endif
}

/** Zero FMatrixArray
 *
 * \param M    FMatrixArray
 *
 * \post matrix memory freed
 */
void Zero_FMatrixArray(FMatrixArray *M ) {

  #ifdef FEMLIB_CUDA
  int   msz   = ((int)(M->numMats/_L)+1)*_L*M->NRows*M->NCols; 
  cudaMemset(M->dEnt, 0, msz*sizeof(Real));
  #else
  int   msz   = M->NRows*M->NCols*M->numMats;
  memset(M->Ent, 0, msz*sizeof(Real));
  #endif
}


/** FMatrix vector product
 *
 * \param M     pointer to previously filled FMatrixArray
 * \param b     pointer to previously filled array
 * \param c     pointer to previously allocated array
 *
 * \post        array c contains the result of the matrix vector product
 */
void FM_PX_Vec(FMatrixArray *M, Real *b, Real *c) {

   const int nmats = M->numMats;
   const int rows  = M->NRows;
   const int cols  = M->NCols;
   const int matSize = rows*cols;
   Real *in  = b;
   Real *out = c;

	#ifdef FEMLIB_CUDA
	// device code
	Real *ent = M->dEnt;
   _device_FM_PX_Vec(ent, rows, cols, nmats, in, out);
	#else
   // host code
   Real *ent = M->Ent;
   for (int k=0; k<nmats; k++, ent+=matSize, in+=cols, out+=cols) {
   	for (int i = 0; i < rows; i++) {
      	Real tmp = 0.;
      	for (int j = 0; j < cols; j++)
      		tmp += ent[j+i*cols]*in[j];
      	out[i] = tmp;
		}  
   }
	#endif        
}

/** FMatrix plus FMatrix
 * \param A     pointer to previously filled FMatrixArray
 * \param B     pointer to previously filled FMatrixArray
 * \param C     pointer to previously initialized FMatrixArray
 *
 * \post        pointer C contains the result of the matrices addition
 */
void FM_Plus_FMArray(FMatrixArray *A, FMatrixArray *B, FMatrixArray *C) {

   const int nmats = A->numMats;
   const int rows = A->NRows;
   const int cols = A->NCols;
	const int matSize = rows*cols;

	#ifdef FEMLIB_CUDA
   Real *Ain = A->dEnt;
	Real *Bin = B->dEnt;
   Real *out = C->dEnt;
	
   _device_FM_Plus_FM(Ain, Bin, out, cols, rows, nmats);
	#else
   Real *Ain = A->Ent;
	Real *Bin = B->Ent;
   Real *out = C->Ent;
	
	for (int i = 0; i < nmats*matSize; i++)
	   out[i] = Ain[i] + Bin[i];
	#endif
}

/** FMatrix Determinant
 * \param M     pointer to previously filled FMatrixArray
 * \param det   pointer to previously declared array
 *
 * \post        array det contains the determinant of each FMatrix
 */
void FMatrix_DetArray(FMatrixArray *M, Real *det) {

   const int nmats   = M->numMats;
   const int rows    = M->NRows;
   const int cols    = M->NCols;
	const int matSize = rows*cols;

	#ifdef FEMLIB_CUDA
	//device code
	Real *dEnt = M->dEnt;
   Real *ldet = det;

   _device_FM_Det(dEnt,cols,rows,nmats,ldet); 
	#else
	// host code
	Real *ent  = M->Ent;
   Real *ldet = det;

	if (rows!=cols) {
   	fprintf(stderr,"\nError in %s(); M.NRows != M.NCols\n", __func__ );
   	exit(1);
  	}

	switch (rows) {
   	case 2: {
         for(int i = 0; i < nmats; i++) {
            const Real a11 = ent[i*matSize], a12 = ent[1+i*matSize], a21 = ent[2+i*matSize], a22 = ent[3+i*matSize];
            ldet[i] = a11*a22 - a12*a21;
         }
		}
      break;
    	case 3: {
         for(int i = 0; i < nmats; i++) {
    			const Real a11 = ent[i*matSize],   a12 = ent[1+i*matSize], a13 = ent[2+i*matSize], 
	    					  a21 = ent[3+i*matSize], a22 = ent[4+i*matSize], a23 = ent[5+i*matSize],
		    				  a31 = ent[6+i*matSize], a32 = ent[7+i*matSize], a33 = ent[8+i*matSize];
			   ldet[i] = (a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33;
         }
		}
      break;
    	case 4: {
         for(int i = 0; i < nmats; i++) {
   			const Real a11 =  ent[i*matSize],   a12 =  ent[1+i*matSize], a13 =  ent[2+i*matSize], a14 =  ent[3+i*matSize], 
	   					  a21 =  ent[4+i*matSize], a22 =  ent[5+i*matSize], a23 =  ent[6+i*matSize], a24 =  ent[7+i*matSize], 
		   				  a31 =  ent[8+i*matSize], a32 =  ent[9+i*matSize], a33 = ent[10+i*matSize], a34 = ent[11+i*matSize],
			   			  a41 = ent[12+i*matSize], a42 = ent[13+i*matSize], a43 = ent[14+i*matSize], a44 = ent[15+i*matSize];
			   ldet[i] =  ((a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33)*a44 
				   	     -((a12*a24 - a14*a22)*a31 - (a11*a24 - a14*a21)*a32 + (a11*a22 - a12*a21)*a34)*a43
					        -((a13*a24 - a14*a23)*a32 - (a12*a24 - a14*a22)*a33 + (a12*a23 - a13*a22)*a34)*a41
					        +((a13*a24 - a14*a23)*a31 - (a11*a24 - a14*a21)*a33 + (a11*a23 - a13*a21)*a34)*a42;
         }
    	}
		break;
    	default: {
      	fprintf(stderr,"\nError in  %s(); M.NRows must be = 2,3, or 4...\n", __func__ );
         exit (1);
      }
	}
	#endif
}

/** FMatrix Inversion
 *
 * \param M       pointer to previously filled FMatrixArray
 * \param M_Inv   pointer to previously declared FMatrixArray
 *
 * \post          M_Inv contains the inverse of M:browse confirm wa
 * \note          M is allowed to be equal M_inv
 *
 */
void  FMatrix_InvArray(FMatrixArray *M, FMatrixArray *M_Inv) {

  const int nmats   = M->numMats;
  const int rows    = M->NRows;
  const int cols    = M->NCols;
  const int matSize = rows*cols;

  if (rows!=cols) {
    fprintf(stderr,"\nError in %s(); M.NRows != M.NCols\n", __func__ );
    exit(1);
  }

#ifdef FEMLIB_CUDA
  //device code
  Real *dEnt     = M->dEnt;
  Real *dEnt_Inv = M_Inv->dEnt;

  _device_FM_Inv(dEnt, dEnt_Inv, cols, rows, nmats); 
#else
  // host code
  Real *ent      = M->Ent;
  Real *ent_inv  = M_Inv->Ent;

  switch (rows) {
    case 2: {
      for(int i = 0; i < nmats; i++) {
        const Real a11 = ent[i*matSize], a12 = ent[1+i*matSize], a21 = ent[2+i*matSize], a22 = ent[3+i*matSize];
        Real ldet = a11*a22 - a12*a21;

        if(ldet == 0) {
          fprintf(stderr,"\nError in %s(); Singular Matrix!\n", __func__ );
          exit(1);
        }

        const Real b11 =  a22/ldet, b12 = -a12/ldet,
                   b21 = -a21/ldet, b22 =  a11/ldet;
        ent_inv[i*matSize]   = b11; ent_inv[1+i*matSize] = b12; 
        ent_inv[2+i*matSize] = b21; ent_inv[3+i*matSize] = b22;
      }
    }
    break;
    case 3: {
      for(int i = 0; i < nmats; i++) {
        const Real a11 = ent[i*matSize],   a12 = ent[1+i*matSize], a13 = ent[2+i*matSize], 
	           a21 = ent[3+i*matSize], a22 = ent[4+i*matSize], a23 = ent[5+i*matSize],
		   a31 = ent[6+i*matSize], a32 = ent[7+i*matSize], a33 = ent[8+i*matSize];
        Real ldet = (a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33;

        if(ldet == 0) {
          fprintf(stderr,"\nError in %s(); Singular Matrix!\n", __func__ );
          exit(1);
        }

        const Real b11 = (a22*a33 - a23*a32)/ldet, b12 = (a13*a32 - a12*a33)/ldet, b13 = (a12*a23 - a13*a22)/ldet,
                   b21 = (a23*a31 - a21*a33)/ldet, b22 = (a11*a33 - a13*a31)/ldet, b23 = (a13*a21 - a11*a23)/ldet,
                   b31 = (a21*a32 - a22*a31)/ldet, b32 = (a12*a31 - a11*a32)/ldet, b33 = (a11*a22 - a12*a21)/ldet;
        ent_inv[i*matSize]   = b11; ent_inv[1+i*matSize] = b12; ent_inv[2+i*matSize] = b13;
	ent_inv[3+i*matSize] = b21; ent_inv[4+i*matSize] = b22; ent_inv[5+i*matSize] = b23;
        ent_inv[6+i*matSize] = b31; ent_inv[7+i*matSize] = b32; ent_inv[8+i*matSize] = b33;
      }
    }
    break;
    case 4: {
      for(int i = 0; i < nmats; i++) {
        const Real a11 =  ent[i*matSize],   a12 =  ent[1+i*matSize], a13 =  ent[2+i*matSize], a14 =  ent[3+i*matSize], 
	           a21 =  ent[4+i*matSize], a22 =  ent[5+i*matSize], a23 =  ent[6+i*matSize], a24 =  ent[7+i*matSize], 
		   a31 =  ent[8+i*matSize], a32 =  ent[9+i*matSize], a33 = ent[10+i*matSize], a34 = ent[11+i*matSize],
		   a41 = ent[12+i*matSize], a42 = ent[13+i*matSize], a43 = ent[14+i*matSize], a44 = ent[15+i*matSize];
        Real ldet = ((a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33)*a44 
	           -((a12*a24 - a14*a22)*a31 - (a11*a24 - a14*a21)*a32 + (a11*a22 - a12*a21)*a34)*a43
		   -((a13*a24 - a14*a23)*a32 - (a12*a24 - a14*a22)*a33 + (a12*a23 - a13*a22)*a34)*a41
		   +((a13*a24 - a14*a23)*a31 - (a11*a24 - a14*a21)*a33 + (a11*a23 - a13*a21)*a34)*a42;

        if(ldet == 0) {
          fprintf(stderr,"\nError in %s(); Singular Matrix!\n", __func__ );
          exit(1);
        }

        const Real b11 =  ((a23*a34 - a24*a33)*a42 - (a22*a34 - a24*a32)*a43 + (a22*a33 - a23*a32)*a44)/ldet,
                   b21 = -((a23*a34 - a24*a33)*a41 - (a21*a34 - a24*a31)*a43 + (a21*a33 - a23*a31)*a44)/ldet,
                   b31 =  ((a22*a34 - a24*a32)*a41 - (a21*a34 - a24*a31)*a42 + (a21*a32 - a22*a31)*a44)/ldet,
                   b41 = -((a22*a33 - a23*a32)*a41 - (a21*a33 - a23*a31)*a42 + (a21*a32 - a22*a31)*a43)/ldet,
                   b12 = -((a13*a34 - a14*a33)*a42 - (a12*a34 - a14*a32)*a43 + (a12*a33 - a13*a32)*a44)/ldet,
                   b22 =  ((a13*a34 - a14*a33)*a41 - (a11*a34 - a14*a31)*a43 + (a11*a33 - a13*a31)*a44)/ldet,
                   b32 = -((a12*a34 - a14*a32)*a41 - (a11*a34 - a14*a31)*a42 + (a11*a32 - a12*a31)*a44)/ldet,
                   b42 =  ((a12*a33 - a13*a32)*a41 - (a11*a33 - a13*a31)*a42 + (a11*a32 - a12*a31)*a43)/ldet,
                   b13 =  ((a13*a24 - a14*a23)*a42 - (a12*a24 - a14*a22)*a43 + (a12*a23 - a13*a22)*a44)/ldet,
                   b23 = -((a13*a24 - a14*a23)*a41 - (a11*a24 - a14*a21)*a43 + (a11*a23 - a13*a21)*a44)/ldet,
                   b33 =  ((a12*a24 - a14*a22)*a41 - (a11*a24 - a14*a21)*a42 + (a11*a22 - a12*a21)*a44)/ldet,
                   b43 = -((a12*a23 - a13*a22)*a41 - (a11*a23 - a13*a21)*a42 + (a11*a22 - a12*a21)*a43)/ldet,
                   b14 = -((a13*a24 - a14*a23)*a32 - (a12*a24 - a14*a22)*a33 + (a12*a23 - a13*a22)*a34)/ldet,
                   b24 =  ((a13*a24 - a14*a23)*a31 - (a11*a24 - a14*a21)*a33 + (a11*a23 - a13*a21)*a34)/ldet,
                   b34 = -((a12*a24 - a14*a22)*a31 - (a11*a24 - a14*a21)*a32 + (a11*a22 - a12*a21)*a34)/ldet,
                   b44 =  ((a12*a23 - a13*a22)*a31 - (a11*a23 - a13*a21)*a32 + (a11*a22 - a12*a21)*a33)/ldet;
                     
                   ent_inv[i*matSize]    = b11; ent_inv[1+i*matSize]  = b12; ent_inv[2+i*matSize]  = b13; 
                   ent_inv[3+i*matSize]  = b14; ent_inv[4+i*matSize]  = b21; ent_inv[5+i*matSize]  = b22; 
                   ent_inv[6+i*matSize]  = b23; ent_inv[7+i*matSize]  = b24; ent_inv[8+i*matSize]  = b31; 
                   ent_inv[9+i*matSize]  = b32; ent_inv[10+i*matSize] = b33; ent_inv[11+i*matSize] = b34;
                   ent_inv[12+i*matSize] = b41; ent_inv[13+i*matSize] = b42; ent_inv[14+i*matSize] = b43; 
                   ent_inv[15+i*matSize] = b44;
         }
    }
    break;
    default: {
      fprintf(stderr,"\nError in  %s(); M.NRows must be = 2,3, or 4...\n", __func__ );
      exit (1);
    }
  }
#endif
}


/*
// not important, don't optimize for now
void FM_Times_FM(FMatrixArray *A, FMatrixArray *B, FMatrixArray *C) {

	if (A->NCols != B->NRows || C->NRows != A->NRows || C-> NCols != B->NCols ||
  	    !A->NCols || !A->NRows || !B->NCols) {
		fprintf(stderr,"\nDimension compatibility error in %s()\n", __func__);
		fprintf(stderr,"A->NRows=%d; A->NCols=%d\n"
	        	  "B->NRows=%d; B->NCols=%d\n"
        		  "C->NRows=%d; C->NCols=%d\n\n",
        			A->NRows,A->NCols,B->NRows,B->NCols,C->NRows,C->NCols);
		exit (1);
	}

	// only one FMatrix
	for (int i = 0; i < A->NRows; i++) 
		for (int j = 0; j < B->NCols; j++)
			for (int k = 0; k < A->NCols; k++)
				C->Ent[j+i*A->NRows] += A->Ent[k+i*A->NRows]*B->Ent[j+k*A->NRows]; // only square matrix for now
		
}*/

// not important for now
/*void FM_Times_Scalar(FMatrixArray *A, Real a) {

	// only one FMatrix
	for(int i=0;i<A->NRows*A->NCols;i++)
      A->Ent[i] *= a;
}*/


