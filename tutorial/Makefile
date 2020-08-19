CC       = mpicxx
CFLAGS   = -std=c++14 -g
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

run: tuto
	mpirun -n 2 ./tuto

clean:
	rm -f tuto
	rm -f $(OBJDIR)/*.o
