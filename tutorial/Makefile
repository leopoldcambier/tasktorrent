CC       = mpicxx
CFLAGS   = -std=c++14 -g -DTTOR_MPI
INCLUDE  = -I../
SRCDIR   = ../tasktorrent/src
OBJDIR   = ../build

.PHONY: clean

DEPS := $(wildcard $(SRCDIR)/*.hpp)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJ := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

default: all
all: tuto tuto_large_am

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

tuto: tuto.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE)

tuto_large_am: tuto_large_am.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE)

run: tuto tuto_large_am
	mpirun -n 2 ./tuto && mpirun -n 2 ./tuto_large_am

clean:
	rm -f tuto tuto_large_am
	rm -f $(OBJDIR)/*.o
