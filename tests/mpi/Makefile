CC       = mpicxx
CFLAGS   = --std=c++14 -g
INCLUDE  = -I../src/
SRCDIR   = ../src
OBJDIR   = ../build

.PHONY: clean

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(OBJ)
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDE)

cholesky: cholesky.cpp $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE)


clean:
	rm -f cholesky
	rm -f $(OBJDIR)/*.o
