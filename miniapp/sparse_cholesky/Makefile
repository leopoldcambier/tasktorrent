include Makefile.conf

CC       = mpicxx
# CFLAGS   = -fsanitize=address -fsanitize=undefined -pthread -O1 -g -std=c++14
CFLAGS   = -pthread -O3 -g -std=c++14 -DEIGEN_NO_DEBUG
INCLUDE  += -I../../src/
LIBS     += 
LDFLAGS  += 
SRCDIR   = ../../src
OBJDIR   = ../../build

.PHONY: default all clean

default: all
all: snchol

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

snchol: snchol.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) $(LDFLAGS)

clean:
	rm -f *.o $(OBJDIR)/*.o
	rm -f snchol
