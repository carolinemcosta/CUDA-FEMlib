########## Makefile for FMatrix GPU ##########

include ../my_switches.def # using new flag FEMLIB_CUDA

# define executable
PROGRAM = main

#  define sources
CC_SOURCES = FMatrixGPU.c testFMatrix.c
CU_SOURCES = FMatrixGPU.cu

#  define objects
ifdef FEMLIB_CUDA
#OBJECTS = $(CU_SOURCES:%.cu=%.o) $(CC_SOURCES:%.c=%.o)
OBJECTS = FMatrixKernels.o $(CC_SOURCES:%.c=%.o)
else
OBJECTS = $(CC_SOURCES:%.c=%.o)
endif

#  define directories
CUDA_HOME = /usr/local/cuda
SDK_HOME  = /usr/local/cuda/samples

#  define compilers
CC   = gcc
NVCC = $(CUDA_HOME)/bin/nvcc

#  define libraries
ifdef FEMLIB_CUDA
ccflags   = -g -O3 -std=gnu99 -I$(SDK_HOME)/inc -I$(CUDA_HOME)/include -DFEMLIB_CUDA
nvccflags = -g -O3 -arch sm_20 --ptxas-options=-v -I$(SDK_HOME)/inc -I$(CUDA_HOME)/include -DFEMLIB_CUDA
libs      = -lm -lcuda -lcudart -L$(CUDA_HOME)/lib64 
else
ccflags = -g -O3 -std=gnu99
libs    = -lm 
endif

# define targets
ifdef FEMLIB_CUDA
default: gpu
else
default: cpu
endif

gpu: ${OBJECTS}
	${CC} -g $^ $(libs) -o ${PROGRAM} 

cpu: $(OBJECTS)
	${CC} -g $^ $(libs) -o ${PROGRAM} 

clean:
	rm -f *.o *~ ${PROGRAM}

# define rules
.PRECIOUS: .cu .h .c 
.SUFFIXES: .cu .h .c .o

#.cu.o:
FMatrixKernels.o: FMatrixGPU.cu
	$(NVCC) $(nvccflags) -c -o FMatrixKernels.o FMatrixGPU.cu

.c.o:
	$(CC) $(ccflags) -c $< 

.h.cu:
	@touch $@

doc:
	doxygen

