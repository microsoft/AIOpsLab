#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, int);
	__type(value, int);
	__uint(max_entries, 1);
} err_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, int);
	__type(value, unsigned char);
	__uint(max_entries, 256);
} pid_map SEC(".maps");

SEC("kprobe/")
int prog1(struct pt_regs *ctx)
{
	int pid = bpf_get_current_pid_tgid() & 0xffffffff;
	int key = 0;
	int *err;

	/* If this a target process */
	if (!bpf_map_lookup_elem(&pid_map, &pid))
		return 0;

	/* If we have a errno */
	err = bpf_map_lookup_elem(&err_map, &key);
	if (!err || !*err)
		return 0;

	/* Now inject the error */
	bpf_override_return(ctx, *err);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
