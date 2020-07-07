include Makefile.conf

CFLAGS   += -pthread -O3 -std=c++14 -Wall -Wextra
INCLUDE  += -I../../../ -I../
LIBS     += 
SRCDIR   = ../../../tasktorrent/src
OBJDIR   = ../../../build

.PHONY: clean

DEPS := $(wildcard $(SRCDIR)/*.hpp)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

all: ttor_deps_mpi starpu_deps_stf_mpi

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

ttor_deps_mpi: ttor_deps_mpi.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

starpu_deps_stf_mpi: starpu_deps_stf_mpi.cpp
	$(CC) $(CFLAGS) $(STARPU_FLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

clean:
	rm -f ttor_deps_mpi starpu_deps_stf_mpi
	rm -f $(OBJDIR)/*.o
