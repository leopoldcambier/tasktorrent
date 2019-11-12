include Makefile.conf

INCLUDE  = -I ../../src/
OBJDIR   = ../../build
SRCDIR   = ../../src

.PHONY: default all clean

default: all
all: cholesky ddot_test tests_comms_internals tests_communicator tests_completion

CXX += -pthread

DEPS = $(SRCDIR)/runtime.hpp $(SRCDIR)/util.hpp $(SRCDIR)/communications.hpp $(SRCDIR)/serialization.hpp $(SRCDIR)/views.hpp $(SRCDIR)/apply_functions.hpp $(SRCDIR)/functional_extra.hpp
OBJ  = $(OBJDIR)/communications.o $(OBJDIR)/serialization.o $(OBJDIR)/util.o

# Objects
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CXX) -o $@ -c $< $(CFLAGS) $(INCLUDE)

# Executables
ddot_test: ddot_test.cpp $(OBJ)
	$(CXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) -I$(EIGEN_INCDIR) $(LDFLAGS) -isystem $(GTEST_INCDIR) $(GTEST_LIBDIR)/libgtest.a

cholesky: cholesky.cpp $(OBJ)
	$(CXX) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) -I$(EIGEN_INCDIR) $(LDFLAGS) -isystem $(GTEST_INCDIR) $(GTEST_LIBDIR)/libgtest.a

tests_communicator: tests_communicator.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(INCLUDE) $(LDFLAGS) -isystem $(GTEST_INCDIR) $(GTEST_LIBDIR)/libgtest.a

tests_completion: tests_completion.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(INCLUDE) $(LIBS) $(LDFLAGS) -isystem $(GTEST_INCDIR) $(GTEST_LIBDIR)/libgtest.a	

tests_comms_internals: tests_comms_internals.cpp $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(INCLUDE) $(LIBS) $(LDFLAGS) -isystem $(GTEST_INCDIR) $(GTEST_LIBDIR)/libgtest.a	

clean:
	-rm -f *.o *.log $(OBJDIR)/*.o
	-rm -f cholesky ddot_test tests_comms_internals tests_communicator tests_completion
