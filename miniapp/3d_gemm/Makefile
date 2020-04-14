include Makefile.conf

CC       = mpicxx
CFLAGS   += -pthread -O3 -g -std=c++14 -DEIGEN_NO_DEBUG -DEIGEN_USE_LAPACKE
INCLUDE  += -I../../src/
LIBS     += 
LDFLAGS  += 
SRCDIR   = ../../src
OBJDIR   = ../../build

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

3d_gemm: 3d_gemm.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LDFLAGS)

clean:
	rm -f 3d_gemm
	rm -f $(OBJDIR)/*.o
