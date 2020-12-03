include Makefile.conf

CFLAGS   += -pthread -O3 -g -std=c++14 -DEIGEN_USE_LAPACKE -DEIGEN_NO_DEBUG
INCLUDE  += -I../../ -I../../external/ ${EIGEN_INC} ${BLAS_INC}
SRCDIR   = ../../tasktorrent/src
OBJDIR   = ../../build

.PHONY: clean

DEPS := $(wildcard $(SRCDIR)/*.hpp)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

default: all
all: 3d_gemm 2d_gemm

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(MPICXX) $(CFLAGS) -o $@ -c $<  $(INCLUDE)

3d_gemm: 3d_gemm.cpp $(OBJ)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(BLAS_PAR_LIBS)

2d_gemm: 2d_gemm.cpp $(OBJ)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(BLAS_SEQ_LIBS)

clean:
	rm -f 3d_gemm 2d_gemm
	rm -f $(OBJDIR)/*.o
