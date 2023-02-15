# ------------------------------------------------------------------------------
#  Makefile 
# ------------------------------------------------------------------------------
ALLOBJS = util.o seeding.o main.o

COMP    = g++ -std=c++11
OPENMP  = -fopenmp -lpthread
OPT     = -w -O3
FLAGS   = -lm -ldl -lnsl -lutil

# ------------------------------------------------------------------------------
#  Compiler with OpenMP
# ------------------------------------------------------------------------------
all:$(ALLOBJS)
	$(COMP) $(OPENMP) $(OPT) -o kpp $(FLAGS) $(ALLOBJS)

%.o: %.cc
	$(COMP) $(OPENMP) -c $(OPT) -o $@ $<

clean:
	-rm $(ALLOBJS) kpp
