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

all: wait cholesky

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

wait: micro_wait.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LDFLAGS)

cholesky: micro_cholesky.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LDFLAGS)


clean:
	rm -f wait cholesky
	rm -f $(OBJDIR)/*.o
