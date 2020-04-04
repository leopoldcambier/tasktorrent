include Makefile.conf

CC       = icpc
CFLAGS   += -pthread -O3 -std=c++14 -DTTOR_SHARED -DNDEBUG
INCLUDE  += -I../../src/
LIBS     += 
LDFLAGS  += -ltbbmalloc_proxy
SRCDIR   = ../../src
OBJDIR   = ../../build

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/serialization.o $(OBJDIR)/util.o

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

mini_benchmarks: mini_benchmarks.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LDFLAGS)

clean:
	rm -f mini_benchmarks
	rm -f $(OBJDIR)/*.o
