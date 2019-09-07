include Makefile.conf

INCLUDE  = -I../../src 
LIBS     =

OBJDIR   = ../../build
SRCDIR   = ../../src

.PHONY: default all clean

default: all
all: tests_serialize tests

CXX  += -pthread

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/serialization.o $(OBJDIR)/util.o

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CXX) -o $@ -c $< $(CFLAGS) $(INCLUDE)

# Executables
tests_serialize: tests_serialize.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) -DTTOR_SHARED $(INCLUDE) $(LIBS) $(LDFLAGS) -isystem $(GTEST_INCDIR) $(GTEST_LIBDIR)/libgtest.a

tests: tests.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) -DTTOR_SHARED $(INCLUDE) $(LIBS) $(LDFLAGS) -I$(EIGEN_INCDIR) -isystem $(GTEST_INCDIR) $(GTEST_LIBDIR)/libgtest.a

run: all
	./tests_serialize && ./tests 64

clean:
	-rm -f *.o *.log $(OBJDIR)/*.o
	-rm -f tests tests_serialize
