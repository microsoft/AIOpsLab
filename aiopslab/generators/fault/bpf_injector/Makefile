CFLAGS += -I./libbpf/include/uapi -I./libbpf/include
LDFLAGS += -L./libbpf/src -lbpf

all: err_inject

%.bpf.o: %.bpf.c
	clang -target bpf -Wall -O2 -g -D__x86_64__ -c $< -o $@
	llvm-strip -g $@

%.skel.h: %.bpf.o
	bpftool gen skeleton $< > $@

err_inject: err_inject.c err_inject.skel.h
	gcc -std=c11 -Wall -O2 $(CFLAGS) $< -o $@ $(LDFLAGS) -static -lbpf -lelf -lz -lzstd

clean:
	rm -rf ebpf err_inject *.o *.skel.h
