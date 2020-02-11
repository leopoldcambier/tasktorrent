CC       = mpicxx
CFLAGS   = -DEIGEN_USE_LAPACKE --std=c++14 -g -O3 -DEIGEN_NO_DEBUG
INCLUDE  = -I../../src/ -I/usr/local/Cellar/eigen/3.3.7/include/eigen3/ -I/usr/local/opt/openblas/include
SRCDIR   = ../../src
OBJDIR   = ../../build
LFLAGS   = -llapack -lblas -lpthread -L/usr/local/opt/openblas/lib

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

cholesky: cholesky.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

clean:
	rm -f cholesky
	rm -f $(OBJDIR)/*.o