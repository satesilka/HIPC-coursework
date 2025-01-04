# CC=gcc
# CFLAGS=-O3 -fopenmp
# LIBFLAGS=-lm

# OBJDIR = obj

# _OBJ = args.o data.o setup.o vtk.o lid-cavity-openMP.o
# OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

# .PHONY: directories

# all: directories lid-cavity-openMP

# obj/%.o: %.c
# 	$(CC) -c -o $@ $< $(CFLAGS) 

# lid-cavity-openMP: $(OBJ)
# 	$(CC) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

# clean:
# 	rm -Rf $(OBJDIR)
# 	rm -f lid-cavity-openMP

# directories: $(OBJDIR)

# $(OBJDIR):
# 	mkdir -p $(OBJDIR)

#to run on my mac
CC=clang
CFLAGS=-O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LIBFLAGS=-L/opt/homebrew/opt/libomp/lib -lomp -lm

OBJDIR = obj

_OBJ = args.o data.o setup.o vtk.o lid-cavity-openMP.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

.PHONY: directories

all: directories lid-cavity-openMP

obj/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

lid-cavity-openMP: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBFLAGS)

clean:
	rm -Rf $(OBJDIR)
	rm -f lid-cavity-openMP

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)
