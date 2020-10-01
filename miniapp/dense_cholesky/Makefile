include Makefile.conf

CFLAGS   += -pthread -O3 -g -std=c++14 -DEIGEN_NO_DEBUG
INCLUDE  += -I../../ -I../../external/
LIBS     += 
SRCDIR   = ../../tasktorrent/src
OBJDIR   = ../../build

.PHONY: clean

DEPS := $(wildcard $(SRCDIR)/*.hpp)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

default: all
all: 2d_cholesky 3d_cholesky

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(MPICXX) $(CFLAGS) -o $@ -c $< $(INCLUDE)

3d_cholesky: 3d_cholesky.cpp $(OBJ)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

2d_cholesky: 2d_cholesky.cpp $(OBJ)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

clean:
	rm -f 3d_cholesky 2d_cholesky
	rm -f $(OBJDIR)/*.o
