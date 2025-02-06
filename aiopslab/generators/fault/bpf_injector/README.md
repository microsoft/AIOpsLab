To build both static libbpf.a and shared libbpf.so:
```bash
$ pushd libbpf/src
$ make
```

Then compile the BPF injector here:
```bash
$ popd
$ make
```

You will get a `err_inject` binary file.