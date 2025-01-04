CC=gcc
CFLAGS=-O3
LIBFLAGS=-lm

OBJDIR = obj

_OBJ = args.o data.o setup.o vtk.o lid-cavity-CUDA.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

.PHONY: directories

all: directories lid-cavity-CUDA

obj/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) 

lid-cavity-CUDA: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

clean:
	rm -Rf $(OBJDIR)
	rm -f lid-cavity-CUDA

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

