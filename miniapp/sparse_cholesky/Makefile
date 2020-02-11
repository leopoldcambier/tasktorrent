include Makefile.conf

INCLUDE  = -I../src/ -I$(EIGEN_INCDIR)  -I$(SCOTCH_INCDIR) -I$(MMIO_INCDIR) -isystem $(GTEST_INCDIR)
LIBS     = -L$(SCOTCH_LIBDIR) $(GTEST_LIBDIR)/libgtest.a
LDFLAGS  = -l scotch -l scotcherr -l tbb
SRCDIR   = ../src
OBJDIR   = ../build

.PHONY: default all clean

default: all
all: snchol

ifeq ($(USE_MKL),1)
    CFLAGS += -DUSE_MKL
endif
ifeq ($(USE_MKL),1)
  INCLUDE += ${MKL_INCFLAGS}
  LIBS    += ${MKL_LIBS} 
  LDFLAGS += ${MKL_LDFLAGS}
else
  INCLUDE += -I${BLAS_INCDIR}
  LIBS    += -L${BLAS_LIBDIR}
  LDFLAGS += -lblas -llapack
endif
CFLAGS += -pthread

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

snchol: snchol.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) $(LDFLAGS)

clean:
	-rm -f *.o *.log ../build/.*o
	-rm -f snchol
