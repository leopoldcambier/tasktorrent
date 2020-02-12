CC       = mpicxx
CFLAGS   = -pthread -O3 -g -std=c++14
INCLUDE  = -I../../src/ -I/usr/local/Cellar/eigen/3.3.7/include/eigen3/ -I/Users/lcambier/Softwares/scotch_6.0.9/include/ -I/usr/local/opt/openblas/include
LIBS     = -L/Users/lcambier/Softwares/scotch_6.0.9/lib/ -L/usr/local/opt/openblas/lib
LDFLAGS  = -lscotch -lscotcherr -lblas -llapack
SRCDIR   = ../../src
OBJDIR   = ../../build

.PHONY: default all clean

default: all
all: snchol

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
