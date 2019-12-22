CC       = g++
CXX		 = mpicxx
CFLAGS   = -DEIGEN_USE_LAPACKE -DTTOR_SHARED --std=c++14 -g -O3 -DEIGEN_NO_DEBUG
CFLAGS1   = -DEIGEN_USE_LAPACKE --std=c++14 -g -O3 -DEIGEN_NO_DEBUG
INCLUDE  = -I../src/
SRCDIR   = ../src
OBJDIR   = ../build
LFLAGS   = -llapack -lblas -lpthread

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/serialization.o $(OBJDIR)/util.o

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CXX) -o $@ -c $< $(CFLAGS) $(INCLUDE)

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(OBJ)
	$(CXX) -o $@ -c $< $(CFLAGS) $(INCLUDE)

tuto: tuto.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LFLAGS)

cholesky: cholesky.cpp $(OBJ)
	$(CXX) $(CFLAGS1) -o $@ $^ $(INCLUDE) $(LFLAGS)

clean:
	rm -f tuto
	rm -f cholesky
	rm -f $(OBJDIR)/*.o
	
