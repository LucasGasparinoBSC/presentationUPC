CC = nvc
CFLAGS = -fast -gpu=cc86,managed,lineinfo -acc -cuda -Minfo=all

all: matmulACC

matmulACC: matmulACC.c
	$(CC) $(CFLAGS) -lnvToolsExt -o $@ $<