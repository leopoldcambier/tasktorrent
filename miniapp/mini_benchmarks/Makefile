include Makefile.conf

CFLAGS   += -pthread -O3 -std=c++14 -DTTOR_SHARED -DNDEBUG
INCLUDE  += -I../../src/
LIBS     += 
LDFLAGS  +=
SRCDIR   = ../../src
OBJDIR   = ../../build

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/serialization.o $(OBJDIR)/util.o

all: wait cholesky chain

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

wait: micro_wait.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) $(LDFLAGS)

cholesky: micro_cholesky.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) $(LDFLAGS)

chain: micro_deps.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) $(LDFLAGS)


clean:
	rm -f wait cholesky chain
	rm -f $(OBJDIR)/*.o
