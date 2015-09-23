# Use makefile in $(LIB_ROOT)

LIB_ROOT = denseinference/lib/

all: 
	make -C $(LIB_ROOT) clean
	make -C $(LIB_ROOT) python
