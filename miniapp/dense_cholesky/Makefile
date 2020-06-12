include Makefile.conf

CFLAGS   += -pthread -O3 -g -std=c++14 -DEIGEN_NO_DEBUG
INCLUDE  += -I../../
LIBS     += 
SRCDIR   = ../../tasktorrent/src
OBJDIR   = ../../build

.PHONY: clean

DEPS := $(wildcard $(SRCDIR)/*.hpp)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

default: all
all: cholesky

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(MPICXX) $(CFLAGS) -o $@ -c $< $(INCLUDE)

cholesky: cholesky.cpp $(OBJ)
	$(MPICXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS)

clean:
	rm -f cholesky
	rm -f $(OBJDIR)/*.o
