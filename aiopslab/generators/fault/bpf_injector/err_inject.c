#include <stdio.h>
#include <stdlib.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "err_inject.skel.h"

int main(int argc, char *argv[])
{
	char *target_syscall = argv[1];
	int err_code = -atoi(argv[2]);
	int nr_pids = argc - 3;
	int *pids = calloc(sizeof(int), nr_pids);
	int key = 0, ret = 0;
	unsigned char value = 0;
	char buf[128] = { 0 };
	struct err_inject_bpf *obj;
	struct bpf_link *link;
	LIBBPF_OPTS(bpf_ksyscall_opts, opts);

	for (int i = 0; i < nr_pids; i++) {
		pids[i] = atoi(argv[i + 3]);
	}

	obj = err_inject_bpf__open_and_load();
	if (!obj) {
		return 1;
	}

	if (bpf_map__update_elem(obj->maps.err_map, &key, sizeof(key),
				 &err_code, sizeof(err_code), 0)) {
		fprintf(stderr, "ERROR: updating err_map failed\n");
		ret = 1;
		goto cleanup;
	}

	for (int i = 0; i < nr_pids; i++) {
		if (bpf_map__update_elem(obj->maps.pid_map, &pids[i],
					 sizeof(pids[i]), &value, sizeof(value),
					 0)) {
			fprintf(stderr, "ERROR: updating pid_map failed\n");
			ret = 1;
			goto cleanup;
		}
	}

	link = bpf_program__attach_ksyscall(obj->progs.prog1, target_syscall,
					    &opts);
	if (libbpf_get_error(link)) {
		fprintf(stderr, "ERROR: bpf_program__attach failed\n");
		link = NULL;
		ret = 1;
		goto cleanup;
	}

	snprintf(buf, sizeof(buf), "/sys/fs/bpf/err_inject-%s", target_syscall);
	bpf_link__pin(link, buf);
	bpf_link__destroy(link);

cleanup:
	err_inject_bpf__destroy(obj);
	return ret;
}
