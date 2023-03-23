CC = nvc
CFLAGS = -fast -gpu=cc61,managed,lineinfo -acc -cuda -Minfo=all

all: matmulACC

matmulACC: matmulACC.c
	$(CC) $(CFLAGS) -lnvToolsExt -o $@ $<
