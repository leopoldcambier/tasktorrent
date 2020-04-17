include Makefile.conf

CFLAGS   = -pthread -O3 -g -std=c++14 -DEIGEN_NO_DEBUG ${USE_MKL}
INCLUDE  += -I../../src/
LIBS     += 
SRCDIR   = ../../src
OBJDIR   = ../../build

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(MPICXX) $(CFLAGS) -o $@ -c $< $(INCLUDE)

cholesky: cholesky.cpp $(OBJ)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

clean:
	rm -f cholesky
	rm -f $(OBJDIR)/*.o
