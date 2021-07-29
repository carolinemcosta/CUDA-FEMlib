#include "FMatrixGPU.h"
#include <time.h>
#include <sys/time.h>


int main() {

   FMatrixArray M, M_inv, B, C;
   Real *det;
   Real *b, *c;
   const int ls = 4,
             nm = 1024*128; // # matrices
   double st_prd, st_pls, st_det, st_inv, 
          et_prd, et_pls, et_det, et_inv;

   #ifdef FEMLIB_CUDA
   Real *b_str, *c_str, // strided matrices on host
        *db, *dc;       // strided matrices on device

   Real *dDet, *det_str;
   #endif

   // allocate memory on host
   Construct_FMatrixArray(&M, ls, ls, nm);
   Construct_FMatrixArray(&M_inv, ls, ls, nm);
   Construct_FMatrixArray(&B, ls, ls, nm);
   Construct_FMatrixArray(&C, ls, ls, nm);

   det = calloc(nm,sizeof(Real));
   b = calloc(nm*ls,sizeof(Real));
   c = calloc(nm*ls,sizeof(Real));

	// set data on host
   for(int i = 0; i < nm; i++) {
      int idxm = i*ls*ls;
      int idxb = i*ls;
      M.Ent[idxm+0] = idxm+10; M.Ent[idxm+1] = idxm+11; M.Ent[idxm+2] = idxm+2;
      M.Ent[idxm+3] = idxm+3; M.Ent[idxm+4] = idxm+4; M.Ent[idxm+5] = idxm+5;
      M.Ent[idxm+6] = idxm+6; M.Ent[idxm+7] = idxm+7; M.Ent[idxm+8] = idxm+8;
      M.Ent[idxm+9] = idxm+9; M.Ent[idxm+10] = idxm+10; M.Ent[idxm+11] = idxm+11;
      M.Ent[idxm+12] = idxm+12; M.Ent[idxm+13] = idxm+13; M.Ent[idxm+14] = idxm+14;
      M.Ent[idxm+15] = idxm+20;

      M_inv.Ent[idxm+0] = 0; M_inv.Ent[idxm+1] = 0; M_inv.Ent[idxm+2] = 0;
      M_inv.Ent[idxm+3] = 0; M_inv.Ent[idxm+4] = 0; M_inv.Ent[idxm+5] = 0;
      M_inv.Ent[idxm+6] = 0; M_inv.Ent[idxm+7] = 0; M_inv.Ent[idxm+8] = 0;
      M_inv.Ent[idxm+9] = 0; M_inv.Ent[idxm+10] = 0; M_inv.Ent[idxm+11] = 0;
      M_inv.Ent[idxm+12] = 0; M_inv.Ent[idxm+13] = 0; M_inv.Ent[idxm+14] = 0;
      M_inv.Ent[idxm+15] = 0;

      B.Ent[idxm+0] = idxm+1; B.Ent[idxm+1] = idxm+1; B.Ent[idxm+2] = idxm+2;
      B.Ent[idxm+3] = idxm+3; B.Ent[idxm+4] = idxm+4; B.Ent[idxm+5] = idxm+5;
      B.Ent[idxm+6] = idxm+6; B.Ent[idxm+7] = idxm+7; B.Ent[idxm+8] = idxm+8;
      B.Ent[idxm+9] = idxm+9; B.Ent[idxm+10] = idxm+10; B.Ent[idxm+11] = idxm+11;
      B.Ent[idxm+12] = idxm+12; B.Ent[idxm+13] = idxm+13; B.Ent[idxm+14] = idxm+14;
      B.Ent[idxm+15] = idxm+20;

      C.Ent[idxm+0] = 0; C.Ent[idxm+1] = 0; C.Ent[idxm+2] = 0;
      C.Ent[idxm+3] = 0; C.Ent[idxm+4] = 0; C.Ent[idxm+5] = 0;
      C.Ent[idxm+6] = 0; C.Ent[idxm+7] = 0; C.Ent[idxm+8] = 0;
      C.Ent[idxm+9] = idxm+9; C.Ent[idxm+10] = idxm+10; C.Ent[idxm+11] = idxm+11;
      C.Ent[idxm+12] = idxm+12; C.Ent[idxm+13] = idxm+13; C.Ent[idxm+14] = idxm+14;
      C.Ent[idxm+15] = idxm+20;

      b[idxb+0] = 1; b[idxb+1] = 2, b[idxb+2] = 3; b[idxb+3] = 4;
	}

   #ifdef FEMLIB_CUDA
   // convert to strided matrix and copy to device
   int size_str = ((int)(nm/_L)+1)*_L*ls*ls*sizeof(Real);
   Convert_Array(M.Ent, M.Ent_str, nm, ls*ls, convertToStridedArray);
   cudaMemcpy(M.dEnt, M.Ent_str, size_str, cudaMemcpyHostToDevice);

   Convert_Array(B.Ent, B.Ent_str, nm, ls*ls, convertToStridedArray);
   cudaMemcpy(B.dEnt, B.Ent_str, size_str, cudaMemcpyHostToDevice);

   Convert_Array(C.Ent, C.Ent_str, nm, ls*ls, convertToStridedArray); // not needed
   cudaMemcpy(C.dEnt, C.Ent_str, size_str, cudaMemcpyHostToDevice);

   // set b
   b_str = calloc(_L*nm*ls,sizeof(Real));
   Convert_Array(b, b_str, nm, ls, convertToStridedArray);
   cudaMalloc((void**)&db, _L*nm*ls*sizeof(Real));
   cudaMemcpy(db, b_str, _L*nm*ls*sizeof(Real), cudaMemcpyHostToDevice);

   // set c
   c_str = calloc(_L*nm*ls,sizeof(Real));
   Convert_Array(c, c_str, nm, ls, convertToStridedArray);
   cudaMalloc((void**)&dc, _L*nm*ls*sizeof(Real));

   // set det
   int size_det = ((int)(nm/_L)+1)*_L;
   det_str = calloc(size_det,sizeof(Real));
   Convert_Array(det, det_str, nm, 1, convertToStridedArray);
   cudaMalloc((void**)&dDet, size_det*sizeof(Real));


   // call functions
   st_prd = clock();
   for(int i = 0; i < 10000; i++)
      FM_PX_Vec(&M, db, dc);
   cudaThreadSynchronize();
   et_prd = (clock() - st_prd)/CLOCKS_PER_SEC;

   st_pls = clock();
   for(int i = 0; i < 10000; i++)
      FM_Plus_FMArray(&M, &B, &C);
   cudaThreadSynchronize();
   et_pls = (clock() - st_pls)/CLOCKS_PER_SEC;

   st_det = clock();
   for(int i = 0; i < 10000; i++)
      FMatrix_DetArray(&M,dDet);
   cudaThreadSynchronize();
   et_det = (clock() - st_det)/CLOCKS_PER_SEC;

   st_inv = clock();
   for(int i = 0; i < 10000; i++)
      FMatrix_InvArray(&M,&M_inv);
   cudaThreadSynchronize();
   et_inv = (clock() - st_inv)/CLOCKS_PER_SEC;


   // copy data from device
   // copy c
   cudaMemcpy(c_str, dc, _L*nm*ls*sizeof(Real), cudaMemcpyDeviceToHost);
   Convert_Array(c, c_str, nm, ls, convertToNonStridedArray);

   // copy C
   cudaMemcpy(C.Ent_str, C.dEnt, size_str, cudaMemcpyDeviceToHost);
   Convert_Array(C.Ent, C.Ent_str, nm, ls*ls, convertToNonStridedArray);

   // copy det
   cudaMemcpy(det_str, dDet, size_det*sizeof(Real), cudaMemcpyDeviceToHost);
   Convert_Array(det, det_str, nm, 1, convertToNonStridedArray);

   // copy M_inv
   cudaMemcpy(M_inv.Ent_str, M_inv.dEnt, size_str, cudaMemcpyDeviceToHost);
   Convert_Array(M_inv.Ent, M_inv.Ent_str, nm, ls*ls, convertToNonStridedArray);

   #else
   st_prd = clock();
   for(int i = 0; i < 10000; i++)
      FM_PX_Vec(&M, b, c);
   et_prd = (clock() - st_prd)/CLOCKS_PER_SEC;

   st_pls = clock();
   for(int i = 0; i < 10000; i++)
      FM_Plus_FMArray(&M, &B, &C);
   et_pls = (clock() - st_pls)/CLOCKS_PER_SEC;

   st_det = clock();
   for(int i = 0; i < 10000; i++)
      FMatrix_DetArray(&M, det);
   et_det = (clock() - st_det)/CLOCKS_PER_SEC;

   st_inv = clock();
   for(int i = 0; i < 10000; i++)
      FMatrix_InvArray(&M,&M_inv);
   et_inv = (clock() - st_inv)/CLOCKS_PER_SEC;

   #endif
	
   // print results of last matrix
   printf("#### Matrix x Vector ####\n\n");
   printf("M\n");
   for(int i = 0; i < ls; i++) {
      for(int j = (nm-1)*ls*ls; j < (nm-1)*ls*ls+ls; j++)
         printf("%f\t", M.Ent[j+i*ls]);
      printf("\n");
   }
   printf("\nb\n");
   for(int i = (nm-1)*ls; i < (nm-1)*ls+ls; i++)
      printf("%f\t", b[i]);

   printf("\nc\n");
   for(int i = (nm-1)*ls; i < (nm-1)*ls+ls; i++)
      printf("%f\t", c[i]);

   printf("\n\n#### Matrix + Matrix ####\n\n");
   printf("A\n");
   for(int i = 0; i < ls; i++) {
      for(int j = (nm-1)*ls*ls; j < (nm-1)*ls*ls+ls; j++)
         printf("%f\t", M.Ent[j]);
      printf("\n");
   }

   printf("\nB\n");
   for(int i = 0; i < ls; i++) {
      for(int j = (nm-1)*ls*ls; j < (nm-1)*ls*ls+ls; j++)
         printf("%f\t", B.Ent[j]);
      printf("\n");
   }

   printf("\nC\n");
   for(int i = 0; i < ls; i++) {
      for(int j = (nm-1)*ls*ls; j < (nm-1)*ls*ls+ls; j++)
         printf("%f\t", C.Ent[j]);
      printf("\n");
   }


   printf("\n\n#### Determinant ####\n\n");
   printf("M\n");
   for(int i = 0; i < ls; i++) {
      for(int j = (nm-1)*ls*ls; j < (nm-1)*ls*ls+ls; j++)
         printf("%f\t", M.Ent[j+i*ls]);
      printf("\n");
   }
   printf("\ndet = %f\n", det[nm-1]);

   printf("\n\n#### Inverse ####\n\n");
   printf("M\n");
   for(int i = 0; i < ls; i++) {
      for(int j = (nm-1)*ls*ls; j < (nm-1)*ls*ls+ls; j++)
         printf("%f\t", M.Ent[j+i*ls]);
      printf("\n");
   }
   printf("\nM_inv\n");
   for(int i = 0; i < ls; i++) {
      for(int j = (nm-1)*ls*ls; j < (nm-1)*ls*ls+ls; j++)
         printf("%f\t", M_inv.Ent[j+i*ls]);
      printf("\n");
   }


   // print elapsed time
   printf("\n\n#### Elapsed times ####\n\n");
   printf("Matrix x Vector: %f seconds\n", et_prd/10000.0);
   printf("Matrix + Matrix: %f seconds\n", et_pls/10000.0);
   printf("Determinant: %f seconds\n", et_det/10000.0);
   printf("Inverse: %f seconds\n", et_inv/10000.0);


   // cleanup
   Destroy_FMatrixArray(&M);
   Destroy_FMatrixArray(&M_inv);
   Destroy_FMatrixArray(&B);
   Destroy_FMatrixArray(&C);
   free(b);
   free(c);
   free(det);

   #ifdef FEMLIB_CUDA
   free(b_str);
   free(c_str);
   cudaFree(db);
   cudaFree(dc);
   free(det_str);
   cudaFree(dDet);
   #endif

   return 0;
}
