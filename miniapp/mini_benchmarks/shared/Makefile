include Makefile.conf

CFLAGS   += -pthread -O3 -std=c++14 -DTTOR_SHARED -DNDEBUG -Wall -Wextra
INCLUDE  += -I../../../ -I../
LIBS     += 
SRCDIR   = ../../../tasktorrent/src
OBJDIR   = ../../../build

.PHONY: clean

DEPS := $(wildcard $(SRCDIR)/*.hpp)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

all: ttor_wait ttor_cholesky ttor_deps omp_wait starpu_wait starpu_wait_stf starpu_deps starpu_deps_stf

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

ttor_wait: ttor_wait.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

ttor_cholesky: ttor_cholesky.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

ttor_deps: ttor_deps.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

omp_wait: omp_wait.cpp
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

starpu_wait: starpu_wait.cpp
	$(CC) $(CFLAGS) $(STARPU_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

starpu_wait_stf: starpu_wait_stf.cpp
	$(CC) $(CFLAGS) $(STARPU_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

starpu_deps: starpu_deps.cpp
	$(CC) $(CFLAGS) $(STARPU_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

starpu_deps_stf: starpu_deps_stf.cpp
	$(CC) $(CFLAGS) $(STARPU_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

clean:
	rm -f ttor_wait ttor_cholesky ttor_deps omp_wait starpu_wait starpu_wait_stf starpu_deps starpu_deps_stf
	rm -f $(OBJDIR)/*.o
