#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <limits.h>
#include <linux/audit.h>
#include <linux/capability.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sched.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_TRACKED_PROCS 8192
#define MAX_DEVICE_BINDS 32
#define QUIESCE_TIMEOUT_MS 5000ULL
#define GROUP_DRAIN_TIMEOUT_MS 5000ULL

#ifndef SECCOMP_RET_KILL_PROCESS
#define SECCOMP_RET_KILL_PROCESS SECCOMP_RET_KILL
#endif

#ifndef PTRACE_O_EXITKILL
#define PTRACE_O_EXITKILL (1 << 20)
#endif

#ifndef PTRACE_O_TRACESECCOMP
#define PTRACE_O_TRACESECCOMP 0x00000080
#endif

#ifndef PTRACE_EVENT_SECCOMP
#define PTRACE_EVENT_SECCOMP 7
#endif

#ifndef PTRACE_SEIZE
#define PTRACE_SEIZE 0x4206
#endif

#ifndef PTRACE_INTERRUPT
#define PTRACE_INTERRUPT 0x4207
#endif

#ifndef PTRACE_EVENT_STOP
#define PTRACE_EVENT_STOP 128
#endif

#ifndef AT_RECURSIVE
#define AT_RECURSIVE 0x8000
#endif

#ifndef MOUNT_ATTR_RDONLY
#define MOUNT_ATTR_RDONLY 0x00000001
#endif

#ifndef __X32_SYSCALL_BIT
#define __X32_SYSCALL_BIT 0x40000000U
#endif

struct device_bind {
    int fd;
    char target[PATH_MAX];
};

struct config {
    const char *run_dir;
    const char *command;
    const char *gpu_uuid;
    int public_fd;
    int rootfs_fd;
    int workspace_fd;
    int submission_fd;
    int memory_fd;
    int tmp_fd;
    int shm_fd;
    int quota_fd;
    int stats_fd;
    int start_fd;
    uid_t host_uid;
    gid_t host_gid;
    rlim_t max_processes;
    rlim_t max_open_files;
    rlim_t max_file_bytes;
    struct device_bind devices[MAX_DEVICE_BINDS];
    size_t device_count;
};

struct proc_state {
    pid_t pid;
    pid_t tgid;
    int active;
    int io_collected;
    int stopped;
    int awaiting_stop;
    int interrupt_pending;
    int waiting_mutation;
    int in_mutation;
    int group_exit_expected;
    int group_exit_stop_seen;
};

enum group_mutation_kind {
    GROUP_MUTATION_NONE = 0,
    GROUP_MUTATION_EXEC,
    GROUP_MUTATION_EXIT_GROUP,
};

struct group_mutation {
    enum group_mutation_kind kind;
    pid_t owner_pid;
    pid_t tgid;
    pid_t survivor_pid;
    unsigned long long deadline_ms;
};

struct trace_stats {
    int exit_code;
    int security_violation;
    int background_process;
    int file_limit;
    int processes_started;
    int active;
    int process_peak;
    unsigned long long bytes_read;
    unsigned long long bytes_written;
    unsigned long long writable_bytes_high_water;
    unsigned long long writable_inodes_high_water;
};

static void __attribute__((noreturn)) die(const char *message) {
    int saved = errno;
    dprintf(STDERR_FILENO, "mlebench supervisor failure: %s: %s\n",
            message, strerror(saved));
    _exit(125);
}

static void arm_parent_death_signal(pid_t expected_parent) {
    if (expected_parent <= 1 || getppid() != expected_parent) {
        errno = ESRCH;
        die("parent identity changed before PR_SET_PDEATHSIG");
    }
    if (prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0) != 0)
        die("set PR_SET_PDEATHSIG");
    if (getppid() != expected_parent) {
        errno = ESRCH;
        die("parent identity changed after PR_SET_PDEATHSIG");
    }
}

static void path_join(char *out, size_t size,
                      const char *left, const char *right) {
    const char *separator = right[0] == '/' ? "" : "/";
    if (snprintf(out, size, "%s%s%s", left, separator, right) >= (int)size) {
        errno = ENAMETOOLONG;
        die("path too long");
    }
}

static int mkdir_one(const char *path, mode_t mode) {
    if (mkdir(path, mode) == 0 || errno == EEXIST) return 0;
    return -1;
}

static void mkdir_path(const char *path, mode_t mode) {
    char buffer[PATH_MAX];
    if (snprintf(buffer, sizeof(buffer), "%s", path) >= (int)sizeof(buffer)) {
        errno = ENAMETOOLONG;
        die("mkdir path");
    }
    for (char *cursor = buffer + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir_one(buffer, mode) != 0) die("mkdir parent");
        *cursor = '/';
    }
    if (mkdir_one(buffer, mode) != 0) die("mkdir leaf");
}

static int parse_nonnegative(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || !end || *end || parsed < 0 || parsed > INT_MAX) {
        errno = EINVAL;
        die(label);
    }
    return (int)parsed;
}

static rlim_t parse_limit(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || !end || *end || parsed == 0) {
        errno = EINVAL;
        die(label);
    }
    return (rlim_t)parsed;
}

static int safe_device_target(const char *target) {
    if (strncmp(target, "/dev/nvidia", 11) != 0 || strstr(target, ".."))
        return 0;
    for (const char *cursor = target; *cursor; cursor++) {
        char c = *cursor;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '/' || c == '-' ||
              c == '_' || c == '.')) return 0;
    }
    return 1;
}

static void parse_device(struct config *cfg, const char *value) {
    if (cfg->device_count >= MAX_DEVICE_BINDS) {
        errno = E2BIG;
        die("too many device binds");
    }
    const char *separator = strchr(value, ':');
    if (!separator || separator == value || !safe_device_target(separator + 1)) {
        errno = EINVAL;
        die("invalid device bind");
    }
    char descriptor[32];
    size_t length = (size_t)(separator - value);
    if (length >= sizeof(descriptor)) {
        errno = EINVAL;
        die("device descriptor too long");
    }
    memcpy(descriptor, value, length);
    descriptor[length] = '\0';
    struct device_bind *item = &cfg->devices[cfg->device_count++];
    item->fd = parse_nonnegative(descriptor, "invalid device descriptor");
    if (snprintf(item->target, sizeof(item->target), "%s", separator + 1) >=
        (int)sizeof(item->target)) {
        errno = ENAMETOOLONG;
        die("device target too long");
    }
}

static void parse_config(int argc, char **argv, struct config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->rootfs_fd = cfg->public_fd = cfg->workspace_fd = -1;
    cfg->submission_fd = -1;
    cfg->memory_fd = cfg->tmp_fd = cfg->shm_fd = cfg->stats_fd = -1;
    cfg->quota_fd = -1;
    cfg->start_fd = -1;
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) {
            errno = EINVAL;
            die("missing argument value");
        }
        const char *name = argv[i++];
        const char *value = argv[i];
        if (strcmp(name, "--rootfs-fd") == 0)
            cfg->rootfs_fd = parse_nonnegative(value, "rootfs fd");
        else if (strcmp(name, "--run-dir") == 0) cfg->run_dir = value;
        else if (strcmp(name, "--command") == 0) cfg->command = value;
        else if (strcmp(name, "--gpu-uuid") == 0) cfg->gpu_uuid = value;
        else if (strcmp(name, "--public-fd") == 0)
            cfg->public_fd = parse_nonnegative(value, "public fd");
        else if (strcmp(name, "--workspace-fd") == 0)
            cfg->workspace_fd = parse_nonnegative(value, "workspace fd");
        else if (strcmp(name, "--submission-fd") == 0)
            cfg->submission_fd = parse_nonnegative(value, "submission fd");
        else if (strcmp(name, "--memory-fd") == 0)
            cfg->memory_fd = parse_nonnegative(value, "memory fd");
        else if (strcmp(name, "--tmp-fd") == 0)
            cfg->tmp_fd = parse_nonnegative(value, "tmp fd");
        else if (strcmp(name, "--shm-fd") == 0)
            cfg->shm_fd = parse_nonnegative(value, "shm fd");
        else if (strcmp(name, "--quota-fd") == 0)
            cfg->quota_fd = parse_nonnegative(value, "quota fd");
        else if (strcmp(name, "--stats-fd") == 0)
            cfg->stats_fd = parse_nonnegative(value, "stats fd");
        else if (strcmp(name, "--start-fd") == 0)
            cfg->start_fd = parse_nonnegative(value, "start fd");
        else if (strcmp(name, "--host-uid") == 0)
            cfg->host_uid = (uid_t)parse_nonnegative(value, "host uid");
        else if (strcmp(name, "--host-gid") == 0)
            cfg->host_gid = (gid_t)parse_nonnegative(value, "host gid");
        else if (strcmp(name, "--max-processes") == 0)
            cfg->max_processes = parse_limit(value, "max processes");
        else if (strcmp(name, "--max-open-files") == 0)
            cfg->max_open_files = parse_limit(value, "max open files");
        else if (strcmp(name, "--max-file-bytes") == 0)
            cfg->max_file_bytes = parse_limit(value, "max file bytes");
        else if (strcmp(name, "--device") == 0) parse_device(cfg, value);
        else {
            errno = EINVAL;
            die("unknown argument");
        }
    }
    if (!cfg->run_dir || !cfg->command || !cfg->gpu_uuid ||
        cfg->rootfs_fd < 0 || cfg->public_fd < 0 || cfg->workspace_fd < 0 ||
        cfg->submission_fd < 0 || cfg->tmp_fd < 0 || cfg->shm_fd < 0 ||
        cfg->quota_fd < 0 || cfg->stats_fd < 0 || cfg->start_fd < 0 ||
        !cfg->max_processes ||
        !cfg->max_open_files ||
        !cfg->max_file_bytes || cfg->device_count < 3) {
        errno = EINVAL;
        die("incomplete arguments");
    }
}

static void bind_directory_fd(int fd, const char *target,
                              int read_only, int noexec) {
    char source[64];
    if (snprintf(source, sizeof(source), "/proc/self/fd/%d", fd) >=
        (int)sizeof(source)) {
        errno = ENAMETOOLONG;
        die("directory fd path");
    }
    if (mount(source, target, NULL, MS_BIND, NULL) != 0)
        die("bind directory fd");
    unsigned long flags = MS_BIND | MS_REMOUNT | MS_NOSUID | MS_NODEV;
    if (read_only) flags |= MS_RDONLY;
    if (noexec) flags |= MS_NOEXEC;
    if (mount(NULL, target, NULL, flags, NULL) != 0)
        die("remount directory fd");
}

static void validate_public_mount_topology(const char *target) {
    FILE *handle = fopen("/proc/self/mountinfo", "re");
    if (!handle) die("open mountinfo for public topology");
    char *line = NULL;
    size_t capacity = 0;
    size_t target_length = strlen(target);
    int exact = 0;
    while (getline(&line, &capacity, handle) >= 0) {
        char mountpoint[PATH_MAX];
        if (sscanf(line, "%*s %*s %*s %*s %4095s", mountpoint) != 1) {
            free(line);
            fclose(handle);
            errno = EPROTO;
            die("parse public mount topology");
        }
        if (strcmp(mountpoint, target) == 0) {
            exact++;
            continue;
        }
        if (strncmp(mountpoint, target, target_length) == 0 &&
            mountpoint[target_length] == '/') {
            free(line);
            fclose(handle);
            errno = EXDEV;
            die("public tree contains a subordinate mount");
        }
    }
    free(line);
    if (fclose(handle) != 0 || exact != 1) {
        errno = EPROTO;
        die("public bind topology drifted");
    }
}

static void make_public_tree_read_only(const char *target) {
    int recursively_sealed = 0;
#ifdef SYS_mount_setattr
    struct mount_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.attr_set = MOUNT_ATTR_RDONLY;
    if (syscall(SYS_mount_setattr, AT_FDCWD, target, AT_RECURSIVE,
                &attributes, sizeof(attributes)) == 0) {
        recursively_sealed = 1;
    } else if (errno != ENOSYS && errno != EINVAL && errno != EOPNOTSUPP) {
        die("recursively seal public tree");
    }
#endif
    if (!recursively_sealed) {
        validate_public_mount_topology(target);
    }
    if (mount(NULL, target, NULL,
              MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV |
                  MS_NOEXEC,
              NULL) != 0)
        die("seal public tree");
    validate_public_mount_topology(target);
}

static void bind_public_directory_fd(int fd, const char *target) {
    char source[64];
    if (snprintf(source, sizeof(source), "/proc/self/fd/%d", fd) >=
        (int)sizeof(source)) {
        errno = ENAMETOOLONG;
        die("public directory fd path");
    }
    if (mount(source, target, NULL, MS_BIND, NULL) != 0)
        die("bind public directory fd");
    validate_public_mount_topology(target);
    make_public_tree_read_only(target);
}

static void bind_device_fd(const struct device_bind *item,
                           const char *merged) {
    char target[PATH_MAX], source[64], parent[PATH_MAX];
    path_join(target, sizeof(target), merged, item->target);
    if (snprintf(parent, sizeof(parent), "%s", target) >= (int)sizeof(parent))
        die("device parent path");
    char *slash = strrchr(parent, '/');
    if (!slash) {
        errno = EINVAL;
        die("device parent");
    }
    *slash = '\0';
    mkdir_path(parent, 0755);
    int target_fd = open(target, O_CREAT | O_WRONLY | O_CLOEXEC, 0600);
    if (target_fd < 0) die("create device bind target");
    close(target_fd);
    if (snprintf(source, sizeof(source), "/proc/self/fd/%d", item->fd) >=
        (int)sizeof(source)) die("device fd path");
    if (mount(source, target, NULL, MS_BIND, NULL) != 0) die("bind device fd");
    if (mount(NULL, target, NULL, MS_BIND | MS_REMOUNT | MS_NOSUID, NULL) != 0)
        die("remount device fd");
}

static void setup_mounts(const struct config *cfg,
                         char *merged, size_t merged_size) {
    char store[PATH_MAX], upper[PATH_MAX], work[PATH_MAX], lower[64];
    char options[PATH_MAX * 3];
    path_join(store, sizeof(store), cfg->run_dir, "overlay-store");
    if (mkdir_one(store, 0700) != 0) die("mkdir overlay store");
    if (mount("tmpfs", store, "tmpfs", MS_NOSUID | MS_NODEV,
              "size=1073741824,nr_inodes=131072,mode=0700") != 0)
        die("mount overlay store");
    path_join(upper, sizeof(upper), store, "upper");
    path_join(work, sizeof(work), store, "work");
    path_join(merged, merged_size, store, "merged");
    if (mkdir_one(upper, 0755) != 0 || mkdir_one(work, 0700) != 0 ||
        mkdir_one(merged, 0755) != 0) die("mkdir overlay directories");
    if (snprintf(lower, sizeof(lower), "/proc/self/fd/%d", cfg->rootfs_fd) >=
        (int)sizeof(lower)) die("rootfs fd path");
    if (snprintf(options, sizeof(options), "lowerdir=%s,upperdir=%s,workdir=%s",
                 lower, upper, work) >= (int)sizeof(options))
        die("overlay options");
    if (mount("overlay", merged, "overlay", MS_NOSUID | MS_NODEV, options) != 0)
        die("mount root overlay");

    char data[PATH_MAX], workspace[PATH_MAX], submission[PATH_MAX];
    char tmp[PATH_MAX], proc[PATH_MAX], dev[PATH_MAX], sysdir[PATH_MAX];
    char run[PATH_MAX], memory[PATH_MAX], oldroot[PATH_MAX], shm[PATH_MAX];
    path_join(data, sizeof(data), merged, "/home/data");
    path_join(workspace, sizeof(workspace), merged, "/home/workspace");
    path_join(submission, sizeof(submission), merged, "/home/submission");
    path_join(tmp, sizeof(tmp), merged, "/tmp");
    path_join(proc, sizeof(proc), merged, "/proc");
    path_join(dev, sizeof(dev), merged, "/dev");
    path_join(sysdir, sizeof(sysdir), merged, "/sys");
    path_join(run, sizeof(run), merged, "/run");
    path_join(memory, sizeof(memory), merged, "/run/amg_memory");
    path_join(oldroot, sizeof(oldroot), merged, "/.oldroot");
    mkdir_path(data, 0755);
    mkdir_path(workspace, 0755);
    mkdir_path(submission, 0755);
    mkdir_path(tmp, 01777);
    mkdir_path(proc, 0555);
    mkdir_path(dev, 0755);
    mkdir_path(sysdir, 0555);
    mkdir_path(run, 0755);
    mkdir_path(oldroot, 0700);
    bind_public_directory_fd(cfg->public_fd, data);
    bind_directory_fd(cfg->workspace_fd, workspace, 0, 0);
    bind_directory_fd(cfg->submission_fd, submission, 0, 0);
    bind_directory_fd(cfg->tmp_fd, tmp, 0, 0);

    if (mount("tmpfs", dev, "tmpfs", MS_NOSUID,
              "size=8388608,nr_inodes=4096,mode=0755") != 0)
        die("mount minimal dev");
    struct {
        const char *name;
        mode_t mode;
        unsigned int major_no;
        unsigned int minor_no;
    } basics[] = {
        {"null", S_IFCHR | 0666, 1, 3},
        {"zero", S_IFCHR | 0666, 1, 5},
        {"random", S_IFCHR | 0666, 1, 8},
        {"urandom", S_IFCHR | 0666, 1, 9},
    };
    for (size_t i = 0; i < sizeof(basics) / sizeof(basics[0]); i++) {
        char node[PATH_MAX];
        path_join(node, sizeof(node), dev, basics[i].name);
        if (mknod(node, basics[i].mode,
                  makedev(basics[i].major_no, basics[i].minor_no)) != 0)
            die("create minimal device");
        if (chmod(node, basics[i].mode & 07777) != 0)
            die("chmod minimal device");
    }
    path_join(shm, sizeof(shm), dev, "shm");
    mkdir_path(shm, 01777);
    bind_directory_fd(cfg->shm_fd, shm, 0, 0);
    for (size_t i = 0; i < cfg->device_count; i++)
        bind_device_fd(&cfg->devices[i], merged);
    char fd_link[PATH_MAX], stdin_link[PATH_MAX], stdout_link[PATH_MAX];
    char stderr_link[PATH_MAX];
    path_join(fd_link, sizeof(fd_link), dev, "fd");
    path_join(stdin_link, sizeof(stdin_link), dev, "stdin");
    path_join(stdout_link, sizeof(stdout_link), dev, "stdout");
    path_join(stderr_link, sizeof(stderr_link), dev, "stderr");
    if (symlink("/proc/self/fd", fd_link) != 0 ||
        symlink("/proc/self/fd/0", stdin_link) != 0 ||
        symlink("/proc/self/fd/1", stdout_link) != 0 ||
        symlink("/proc/self/fd/2", stderr_link) != 0)
        die("create device fd links");
    if (chmod(dev, 0555) != 0) die("seal device directory");

    if (mount("proc", proc, "proc", MS_NOEXEC | MS_NOSUID | MS_NODEV,
              "hidepid=2") != 0) die("mount proc");
    if (mount("tmpfs", sysdir, "tmpfs", MS_NOEXEC | MS_NOSUID | MS_NODEV,
              "size=65536,nr_inodes=64,mode=0555") != 0) die("mount empty sys");
    if (mount(NULL, sysdir, NULL,
              MS_REMOUNT | MS_RDONLY | MS_NOEXEC | MS_NOSUID | MS_NODEV,
              NULL) != 0) die("seal empty sys");
    if (mount("tmpfs", run, "tmpfs", MS_NOSUID | MS_NODEV,
              "size=16777216,nr_inodes=4096,mode=0755") != 0)
        die("mount private run");
    if (cfg->memory_fd >= 0) {
        mkdir_path(memory, 0755);
        bind_directory_fd(cfg->memory_fd, memory, 0, 0);
    }
    if (mount(NULL, run, NULL, MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV,
              NULL) != 0) die("seal private run");
    if (mount(NULL, merged, NULL,
              MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV, NULL) != 0)
        die("seal root filesystem");
}

static void enter_root(const char *merged) {
    if (chdir(merged) != 0) die("chdir new root");
    if (syscall(SYS_pivot_root, ".", ".oldroot") != 0) die("pivot root");
    if (chdir("/") != 0) die("chdir sandbox root");
    if (umount2("/.oldroot", MNT_DETACH) != 0) die("detach old root");
}

static void write_all(int fd, const char *value) {
    size_t left = strlen(value);
    while (left) {
        ssize_t size = write(fd, value, left);
        if (size < 0) {
            if (errno == EINTR) continue;
            die("write synchronization");
        }
        value += size;
        left -= (size_t)size;
    }
}

static void write_map(const char *path, const char *value, int optional) {
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        if (optional && errno == ENOENT) return;
        die("open namespace map");
    }
    write_all(fd, value);
    close(fd);
}

static void install_user_map(const char *merged, pid_t child,
                             uid_t host_uid, gid_t host_gid) {
    char path[PATH_MAX], value[128];
    if (snprintf(path, sizeof(path), "%s/proc/%d/setgroups", merged, child) >=
        (int)sizeof(path)) die("setgroups map path");
    write_map(path, "deny\n", 1);
    if (snprintf(path, sizeof(path), "%s/proc/%d/uid_map", merged, child) >=
        (int)sizeof(path)) die("uid map path");
    if (snprintf(value, sizeof(value), "0 0 1\n1000 %u 1\n",
                 (unsigned int)host_uid) >= (int)sizeof(value)) die("uid map");
    write_map(path, value, 0);
    if (snprintf(path, sizeof(path), "%s/proc/%d/gid_map", merged, child) >=
        (int)sizeof(path)) die("gid map path");
    if (snprintf(value, sizeof(value), "0 0 1\n1000 %u 1\n",
                 (unsigned int)host_gid) >= (int)sizeof(value)) die("gid map");
    write_map(path, value, 0);
}

#define DENY_SYSCALL(name)                                                   \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_##name, 0, 1),                 \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP)

#define ERRNO_SYSCALL(name, value)                                           \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_##name, 0, 1),                 \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO |                           \
             ((value) & SECCOMP_RET_DATA))

#define TRACE_MUTATION_SYSCALL(name)                                        \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_##name, 0, 1),                 \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE)

static void install_seccomp(void) {
    const uint32_t namespace_flags =
        CLONE_NEWNS | CLONE_NEWCGROUP | CLONE_NEWUTS | CLONE_NEWIPC |
        CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWNET;
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, __X32_SYSCALL_BIT, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_socket, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, args[0])),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_socketpair, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, args[0])),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, nr)),
        DENY_SYSCALL(mount),
        DENY_SYSCALL(umount2),
        DENY_SYSCALL(pivot_root),
        DENY_SYSCALL(ptrace),
        DENY_SYSCALL(unshare),
        DENY_SYSCALL(setns),
        DENY_SYSCALL(bpf),
        DENY_SYSCALL(perf_event_open),
        DENY_SYSCALL(init_module),
        DENY_SYSCALL(finit_module),
        DENY_SYSCALL(delete_module),
        DENY_SYSCALL(kexec_load),
        DENY_SYSCALL(reboot),
        DENY_SYSCALL(swapon),
        DENY_SYSCALL(swapoff),
        DENY_SYSCALL(open_by_handle_at),
        DENY_SYSCALL(name_to_handle_at),
        DENY_SYSCALL(keyctl),
        DENY_SYSCALL(add_key),
        DENY_SYSCALL(request_key),
        DENY_SYSCALL(mknod),
        DENY_SYSCALL(mknodat),
        DENY_SYSCALL(chroot),
        DENY_SYSCALL(setuid),
        DENY_SYSCALL(setgid),
        DENY_SYSCALL(setreuid),
        DENY_SYSCALL(setregid),
        DENY_SYSCALL(setresuid),
        DENY_SYSCALL(setresgid),
        DENY_SYSCALL(setfsuid),
        DENY_SYSCALL(setfsgid),
        DENY_SYSCALL(capset),
        DENY_SYSCALL(seccomp),
#ifdef __NR_userfaultfd
        DENY_SYSCALL(userfaultfd),
#endif
#ifdef __NR_clone3
        ERRNO_SYSCALL(clone3, ENOSYS),
#endif
#ifdef __NR_io_uring_setup
        DENY_SYSCALL(io_uring_setup),
#endif
#ifdef __NR_io_uring_enter
        DENY_SYSCALL(io_uring_enter),
#endif
#ifdef __NR_io_uring_register
        DENY_SYSCALL(io_uring_register),
#endif
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone, 0, 5),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, args[0])),
        BPF_STMT(BPF_ALU | BPF_AND | BPF_K, namespace_flags),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_open, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, args[1])),
        BPF_STMT(BPF_ALU | BPF_AND | BPF_K, O_CREAT | O_TRUNC | O_TMPFILE),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_openat, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, args[2])),
        BPF_STMT(BPF_ALU | BPF_AND | BPF_K, O_CREAT | O_TRUNC | O_TMPFILE),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (uint32_t)offsetof(struct seccomp_data, nr)),
#ifdef __NR_openat2
        TRACE_MUTATION_SYSCALL(openat2),
#endif
        TRACE_MUTATION_SYSCALL(creat),
        TRACE_MUTATION_SYSCALL(close),
#ifdef __NR_close_range
        TRACE_MUTATION_SYSCALL(close_range),
#endif
#ifdef __NR_dup2
        TRACE_MUTATION_SYSCALL(dup2),
#endif
#ifdef __NR_dup3
        TRACE_MUTATION_SYSCALL(dup3),
#endif
        TRACE_MUTATION_SYSCALL(execve),
#ifdef __NR_execveat
        TRACE_MUTATION_SYSCALL(execveat),
#endif
#ifdef __NR_mmap
        TRACE_MUTATION_SYSCALL(mmap),
#endif
#ifdef __NR_munmap
        TRACE_MUTATION_SYSCALL(munmap),
#endif
#ifdef __NR_mremap
        TRACE_MUTATION_SYSCALL(mremap),
#endif
#ifdef __NR_madvise
        TRACE_MUTATION_SYSCALL(madvise),
#endif
#ifdef __NR_process_madvise
        TRACE_MUTATION_SYSCALL(process_madvise),
#endif
        TRACE_MUTATION_SYSCALL(truncate),
        TRACE_MUTATION_SYSCALL(ftruncate),
        TRACE_MUTATION_SYSCALL(fallocate),
        TRACE_MUTATION_SYSCALL(unlink),
        TRACE_MUTATION_SYSCALL(unlinkat),
        TRACE_MUTATION_SYSCALL(rename),
        TRACE_MUTATION_SYSCALL(renameat),
#ifdef __NR_renameat2
        TRACE_MUTATION_SYSCALL(renameat2),
#endif
        TRACE_MUTATION_SYSCALL(mkdir),
        TRACE_MUTATION_SYSCALL(mkdirat),
        TRACE_MUTATION_SYSCALL(rmdir),
        TRACE_MUTATION_SYSCALL(link),
        TRACE_MUTATION_SYSCALL(linkat),
        TRACE_MUTATION_SYSCALL(symlink),
        TRACE_MUTATION_SYSCALL(symlinkat),
        TRACE_MUTATION_SYSCALL(exit),
        TRACE_MUTATION_SYSCALL(exit_group),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
        .filter = filter,
    };
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) die("no new privileges");
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0)
        die("install seccomp");
}

static void set_limit(int resource, rlim_t value) {
    struct rlimit limit = {.rlim_cur = value, .rlim_max = value};
    if (setrlimit(resource, &limit) != 0) die("set resource limit");
}

static void drop_capability_bounding_set(void) {
    for (int cap = 0; cap <= 63; cap++) {
        if (prctl(PR_CAPBSET_DROP, cap, 0, 0, 0) != 0 && errno != EINVAL)
            die("drop capability bound");
    }
}

static void clear_capabilities(void) {
    struct __user_cap_header_struct header;
    struct __user_cap_data_struct data[2];
    memset(&header, 0, sizeof(header));
    memset(data, 0, sizeof(data));
    header.version = _LINUX_CAPABILITY_VERSION_3;
    if (syscall(SYS_capset, &header, data) != 0) die("clear capabilities");
}

static void child_environment(const struct config *cfg) {
    if (clearenv() != 0) die("clear environment");
    const char *pairs[][2] = {
        {"PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
        {"HOME", "/home/workspace"},
        {"TMPDIR", "/tmp"},
        {"TZ", "UTC"},
        {"LC_ALL", "C.UTF-8"},
        {"LANG", "C.UTF-8"},
        {"PYTHONDONTWRITEBYTECODE", "1"},
        {"CUDA_VISIBLE_DEVICES", "0"},
    };
    for (size_t i = 0; i < sizeof(pairs) / sizeof(pairs[0]); i++)
        if (setenv(pairs[i][0], pairs[i][1], 1) != 0) die("set environment");
    if (setenv("NVIDIA_VISIBLE_DEVICES", cfg->gpu_uuid, 1) != 0)
        die("set GPU identity");
}

static void close_untrusted_fds(const struct config *cfg) {
#ifdef __NR_close_range
    if (syscall(__NR_close_range, 3U, ~0U, 0U) == 0) return;
    if (errno != ENOSYS && errno != EINVAL) die("close inherited descriptors");
#endif
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) die("read descriptor limit");
    unsigned long maximum = limit.rlim_max == RLIM_INFINITY
        ? (unsigned long)cfg->max_open_files
        : (unsigned long)limit.rlim_max;
    if (maximum < (unsigned long)cfg->max_open_files)
        maximum = (unsigned long)cfg->max_open_files;
    for (unsigned long fd = 3; fd < maximum; fd++) close((int)fd);
}

static void sandbox_child(const struct config *cfg,
                          int ready_write, int go_read) {
    if (chdir("/home/workspace") != 0) die("enter workspace");
    if (setgroups(0, NULL) != 0) die("clear groups");
    if (unshare(CLONE_NEWUSER) != 0) die("create user namespace");
    write_all(ready_write, "U");
    char token;
    if (read(go_read, &token, 1) != 1) die("wait user map");
    close(ready_write);
    close(go_read);
    drop_capability_bounding_set();
    if (setresgid(1000, 1000, 1000) != 0) die("drop gid");
    if (setresuid(1000, 1000, 1000) != 0) die("drop uid");
    clear_capabilities();
    set_limit(RLIMIT_FSIZE, cfg->max_file_bytes);
    set_limit(RLIMIT_CORE, 0);
    set_limit(RLIMIT_NPROC, cfg->max_processes);
    set_limit(RLIMIT_NOFILE, cfg->max_open_files);
    close(cfg->stats_fd);
    child_environment(cfg);
    install_seccomp();
    close_untrusted_fds(cfg);
    execl("/bin/sh", "sh", "-c", cfg->command, (char *)NULL);
    die("execute shell");
}

static struct proc_state *find_state(struct proc_state *states, pid_t pid) {
    for (int i = 0; i < MAX_TRACKED_PROCS; i++)
        if (states[i].active && states[i].pid == pid) return &states[i];
    return NULL;
}

static pid_t read_tracee_tgid(pid_t pid) {
    char path[128];
    if (snprintf(path, sizeof(path), "/proc/%d/status", pid) >=
        (int)sizeof(path)) {
        errno = ENAMETOOLONG;
        die("build tracee status path");
    }
    FILE *handle = fopen(path, "r");
    if (!handle) die("open tracee status");
    char line[256];
    pid_t tgid = 0;
    while (fgets(line, sizeof(line), handle)) {
        long value;
        char trailing;
        if (sscanf(line, "Tgid:%ld %c", &value, &trailing) == 1 &&
            value > 0 && value <= INT_MAX) {
            tgid = (pid_t)value;
            break;
        }
    }
    if (fclose(handle) != 0) die("close tracee status");
    if (tgid <= 0) {
        errno = EPROTO;
        die("read tracee thread group");
    }
    return tgid;
}

static struct proc_state *add_state(struct proc_state *states, pid_t pid,
                                    struct trace_stats *stats) {
    struct proc_state *existing = find_state(states, pid);
    if (existing) return existing;
    for (int i = 0; i < MAX_TRACKED_PROCS; i++) {
        if (!states[i].active) {
            memset(&states[i], 0, sizeof(states[i]));
            states[i].pid = pid;
            states[i].tgid = read_tracee_tgid(pid);
            states[i].active = 1;
            stats->active++;
            stats->processes_started++;
            if (stats->active > stats->process_peak) stats->process_peak = stats->active;
            return &states[i];
        }
    }
    errno = EOVERFLOW;
    die("trace process table exhausted");
}

static void collect_io(struct proc_state *state, struct trace_stats *stats) {
    if (!state || state->io_collected) return;
    char path[128];
    if (snprintf(path, sizeof(path), "/proc/%d/io", state->pid) >=
        (int)sizeof(path)) return;
    FILE *handle = fopen(path, "r");
    if (!handle) return;
    char key[64];
    unsigned long long value;
    while (fscanf(handle, "%63[^:]: %llu\n", key, &value) == 2) {
        if (strcmp(key, "read_bytes") == 0) stats->bytes_read += value;
        else if (strcmp(key, "write_bytes") == 0) stats->bytes_written += value;
    }
    fclose(handle);
    state->io_collected = 1;
}

static void kill_tracees(struct proc_state *states) {
    kill(-1, SIGKILL);
    for (int i = 0; i < MAX_TRACKED_PROCS; i++) {
        if (!states[i].active || !states[i].stopped) continue;
        if (ptrace(PTRACE_CONT, states[i].pid, NULL,
                   (void *)(long)SIGKILL) == 0) {
            states[i].stopped = 0;
            states[i].interrupt_pending = 0;
            states[i].awaiting_stop = 0;
            states[i].waiting_mutation = 0;
            states[i].in_mutation = 0;
            states[i].group_exit_expected = 0;
            states[i].group_exit_stop_seen = 0;
        }
    }
}

static void sample_writable_high_water(const struct config *cfg,
                                       struct trace_stats *stats) {
    struct statvfs values;
    if (fstatvfs(cfg->quota_fd, &values) != 0)
        die("sample writable high-water");
    unsigned long long used_blocks =
        (unsigned long long)(values.f_blocks - values.f_bfree);
    unsigned long long fragment = (unsigned long long)values.f_frsize;
    if (fragment != 0 && used_blocks > ULLONG_MAX / fragment) {
        errno = EOVERFLOW;
        die("writable high-water overflow");
    }
    unsigned long long bytes = used_blocks * fragment;
    unsigned long long inodes =
        (unsigned long long)(values.f_files - values.f_ffree);
    if (bytes > stats->writable_bytes_high_water)
        stats->writable_bytes_high_water = bytes;
    if (inodes > stats->writable_inodes_high_water)
        stats->writable_inodes_high_water = inodes;
}

static unsigned long long monotonic_milliseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        die("read monotonic clock");
    if ((unsigned long long)value.tv_sec > ULLONG_MAX / 1000ULL) {
        errno = EOVERFLOW;
        die("monotonic clock overflow");
    }
    return (unsigned long long)value.tv_sec * 1000ULL +
           (unsigned long long)value.tv_nsec / 1000000ULL;
}

static int restartable_storage_writer(long syscall_number) {
    switch (syscall_number) {
        case __NR_write:
        case __NR_writev:
        case __NR_pwrite64:
#ifdef __NR_pwritev
        case __NR_pwritev:
#endif
#ifdef __NR_pwritev2
        case __NR_pwritev2:
#endif
#ifdef __NR_sendfile
        case __NR_sendfile:
#endif
#ifdef __NR_copy_file_range
        case __NR_copy_file_range:
#endif
#ifdef __NR_splice
        case __NR_splice:
#endif
#ifdef __NR_process_vm_writev
        case __NR_process_vm_writev:
#endif
            return 1;
        default:
            return 0;
    }
}

static void cancel_interrupted_storage_writer(pid_t pid) {
    struct user_regs_struct registers;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &registers) != 0)
        die("read interrupted tracee registers");
    long long result = (long long)registers.rax;
    long syscall_number = (long)registers.orig_rax;
    if (!restartable_storage_writer(syscall_number) ||
        (result != -512LL && result != -513LL &&
         result != -514LL && result != -516LL))
        return;
    registers.orig_rax = (unsigned long long)-1LL;
    registers.rax = (unsigned long long)-EINTR;
    if (ptrace(PTRACE_SETREGS, pid, NULL, &registers) != 0)
        die("cancel interrupted storage writer");
}

static enum group_mutation_kind group_destructive_mutation(pid_t pid) {
    struct user_regs_struct registers;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &registers) != 0)
        die("read serialized mutation registers");
    long syscall_number = (long)registers.orig_rax;
    if (syscall_number == __NR_execve
#ifdef __NR_execveat
        || syscall_number == __NR_execveat
#endif
    ) return GROUP_MUTATION_EXEC;
    if (syscall_number == __NR_exit_group)
        return GROUP_MUTATION_EXIT_GROUP;
    return GROUP_MUTATION_NONE;
}

static void start_group_destructive_mutation(
        struct proc_state *states, struct proc_state *owner,
        struct group_mutation *group) {
    enum group_mutation_kind kind = group_destructive_mutation(owner->pid);
    if (kind == GROUP_MUTATION_NONE) return;
    if (group->kind != GROUP_MUTATION_NONE || owner->tgid <= 0) {
        errno = EPROTO;
        die("group-destructive mutation state drifted");
    }
    unsigned long long now = monotonic_milliseconds();
    if (now > ULLONG_MAX - GROUP_DRAIN_TIMEOUT_MS) {
        errno = EOVERFLOW;
        die("group-destructive mutation deadline overflow");
    }
    group->kind = kind;
    group->owner_pid = owner->pid;
    group->tgid = owner->tgid;
    group->survivor_pid = 0;
    group->deadline_ms = now + GROUP_DRAIN_TIMEOUT_MS;
    int members = 0;
    for (int i = 0; i < MAX_TRACKED_PROCS; i++) {
        struct proc_state *state = &states[i];
        if (!state->active || state->tgid != group->tgid) continue;
        state->group_exit_expected = 1;
        state->group_exit_stop_seen = 0;
        members++;
    }
    if (!owner->group_exit_expected || members <= 0) {
        errno = EPROTO;
        die("group-destructive mutation membership drifted");
    }
}

static void clear_group_destructive_mutation(
        struct proc_state *states, struct group_mutation *group) {
    for (int i = 0; i < MAX_TRACKED_PROCS; i++) {
        states[i].group_exit_expected = 0;
        states[i].group_exit_stop_seen = 0;
    }
    memset(group, 0, sizeof(*group));
}

static int group_destructive_drain_complete(
        struct proc_state *states, const struct group_mutation *group) {
    if (group->kind == GROUP_MUTATION_NONE) return 1;
    if (group->kind == GROUP_MUTATION_EXEC && group->survivor_pid <= 0)
        return 0;
    for (int i = 0; i < MAX_TRACKED_PROCS; i++)
        if (states[i].active && states[i].group_exit_expected) return 0;
    return 1;
}

static struct proc_state *remap_exec_owner(
        struct proc_state *states, pid_t event_pid,
        struct trace_stats *stats, struct group_mutation *group,
        pid_t *mutation_owner) {
    if (group->kind != GROUP_MUTATION_EXEC) {
        errno = EPROTO;
        die("unexpected exec event outside group mutation");
    }
    unsigned long former_tid = 0;
    if (ptrace(PTRACE_GETEVENTMSG, event_pid, NULL, &former_tid) != 0 ||
        former_tid == 0 || former_tid > INT_MAX) {
        errno = EPROTO;
        die("read former exec thread identity");
    }
    struct proc_state *owner = find_state(states, (pid_t)former_tid);
    if (!owner || !owner->active || !owner->in_mutation ||
        *mutation_owner != (pid_t)former_tid ||
        group->owner_pid != (pid_t)former_tid ||
        owner->tgid != group->tgid) {
        errno = EPROTO;
        die("remap exec owner state drifted");
    }
    if ((pid_t)former_tid != event_pid) {
        struct proc_state *displaced = find_state(states, event_pid);
        if (displaced && displaced != owner) {
            if (!displaced->group_exit_expected ||
                displaced->tgid != group->tgid) {
                errno = EPROTO;
                die("exec displaced an unrelated tracee");
            }
            collect_io(displaced, stats);
            displaced->active = 0;
            displaced->stopped = 0;
            displaced->waiting_mutation = 0;
            displaced->in_mutation = 0;
            displaced->group_exit_expected = 0;
            stats->active--;
        }
        owner->pid = event_pid;
        owner->tgid = event_pid;
        *mutation_owner = event_pid;
        group->owner_pid = event_pid;
    }
    group->survivor_pid = event_pid;
    return owner;
}

static int every_tracee_stopped(struct proc_state *states) {
    for (int i = 0; i < MAX_TRACKED_PROCS; i++)
        if (states[i].active && !states[i].stopped) return 0;
    return 1;
}

static void resume_tracee(struct proc_state *state, int request, int signal_number,
                          const char *label) {
    if (!state || !state->active || !state->stopped) {
        errno = EPROTO;
        die(label);
    }
    if (ptrace(request, state->pid, NULL,
               (void *)(long)signal_number) != 0)
        die(label);
    state->stopped = 0;
    state->interrupt_pending = 0;
    state->awaiting_stop = 0;
}

static int handle_group_exit_stop(
        const struct config *cfg, struct proc_state *state,
        struct trace_stats *stats, const struct group_mutation *group) {
    if (group->kind == GROUP_MUTATION_NONE || !state || !state->active ||
        !state->group_exit_expected || state->tgid != group->tgid)
        return 0;
    if (!state->stopped || state->group_exit_stop_seen) {
        errno = EPROTO;
        die("group exit stop state drifted");
    }
    state->group_exit_stop_seen = 1;
    state->waiting_mutation = 0;
    state->in_mutation = 0;
    sample_writable_high_water(cfg, stats);
    collect_io(state, stats);
    resume_tracee(state, PTRACE_CONT, 0, "drain group exit stop");
    return 1;
}

static void quiesce_tracees(const struct config *cfg,
                            struct proc_state *states,
                            struct proc_state *owner,
                            struct trace_stats *stats) {
    if (!owner || !owner->active || !owner->stopped) {
        errno = EPROTO;
        die("quiescence owner is not stopped");
    }
    for (int i = 0; i < MAX_TRACKED_PROCS; i++) {
        struct proc_state *state = &states[i];
        if (!state->active || state->stopped || state->awaiting_stop) continue;
        if (ptrace(PTRACE_INTERRUPT, state->pid, NULL, NULL) == 0) {
            state->interrupt_pending = 1;
            continue;
        }
        if (errno != ESRCH && errno != EIO)
            die("interrupt tracee for writable high-water sample");
        state->interrupt_pending = 1;
    }

    unsigned long long now = monotonic_milliseconds();
    if (now > ULLONG_MAX - QUIESCE_TIMEOUT_MS) {
        errno = EOVERFLOW;
        die("quiescence deadline overflow");
    }
    unsigned long long deadline = now + QUIESCE_TIMEOUT_MS;
    while (!every_tracee_stopped(states)) {
        int status;
        pid_t pid = waitpid(-1, &status, __WALL | WNOHANG);
        if (pid == 0) {
            if (monotonic_milliseconds() >= deadline) {
                errno = ETIMEDOUT;
                die("bounded tracee quiescence timed out");
            }
            struct timespec pause = {.tv_sec = 0, .tv_nsec = 1000000L};
            while (nanosleep(&pause, &pause) != 0 && errno == EINTR) {}
            continue;
        }
        if (pid < 0) {
            if (errno == EINTR) continue;
            die("wait for tracee quiescence");
        }
        struct proc_state *state = find_state(states, pid);
        if (!state || WIFEXITED(status) || WIFSIGNALED(status) ||
            !WIFSTOPPED(status)) {
            errno = EPROTO;
            die("tracee exit race during quiescence");
        }
        state->stopped = 1;
        int signal_number = WSTOPSIG(status);
        unsigned int event = (unsigned int)status >> 16;
        if (event == PTRACE_EVENT_STOP &&
            (state->interrupt_pending || state->awaiting_stop)) {
            if (state->interrupt_pending)
                cancel_interrupted_storage_writer(pid);
            state->interrupt_pending = 0;
            state->awaiting_stop = 0;
            continue;
        }
        if (event == PTRACE_EVENT_SECCOMP) {
            if (state->waiting_mutation || state->in_mutation) {
                errno = EPROTO;
                die("duplicate mutation stop during quiescence");
            }
            state->interrupt_pending = 0;
            state->waiting_mutation = 1;
            continue;
        }
        if (event == PTRACE_EVENT_FORK || event == PTRACE_EVENT_VFORK ||
            event == PTRACE_EVENT_CLONE) {
            errno = EBUSY;
            die("fork race during writable high-water quiescence");
        }
        if (event == PTRACE_EVENT_EXIT) {
            errno = EBUSY;
            die("exit race during writable high-water quiescence");
        }
        if (event != 0 || signal_number == (SIGTRAP | 0x80)) {
            errno = EPROTO;
            die("unexpected ptrace stop during quiescence");
        }
        errno = EINTR;
        die("signal race during writable high-water quiescence");
    }
    sample_writable_high_water(cfg, stats);
}

static void resume_quiesced_tracees(struct proc_state *states) {
    for (int i = 0; i < MAX_TRACKED_PROCS; i++) {
        struct proc_state *state = &states[i];
        if (!state->active || !state->stopped || state->waiting_mutation ||
            state->in_mutation || state->group_exit_expected)
            continue;
        resume_tracee(state, PTRACE_CONT, 0, "resume quiesced tracee");
    }
}

static void write_stats(const struct config *cfg,
                        const struct trace_stats *stats) {
    dprintf(cfg->stats_fd,
            "{\"schema\":\"mlebench_lite_supervisor_stats_v1\","
            "\"exit_code\":%d,\"security_violation\":%s,"
            "\"background_process\":%s,\"file_limit\":%s,"
            "\"processes_started\":%d,\"process_peak\":%d,"
            "\"bytes_read\":%llu,\"bytes_written\":%llu,"
            "\"writable_bytes_high_water\":%llu,"
            "\"writable_inodes_high_water\":%llu}",
            stats->exit_code,
            stats->security_violation ? "true" : "false",
            stats->background_process ? "true" : "false",
            stats->file_limit ? "true" : "false",
            stats->processes_started, stats->process_peak,
            stats->bytes_read, stats->bytes_written,
            stats->writable_bytes_high_water,
            stats->writable_inodes_high_water);
}

static struct proc_state *next_waiting_mutation(struct proc_state *states) {
    for (int i = 0; i < MAX_TRACKED_PROCS; i++)
        if (states[i].active && states[i].waiting_mutation) return &states[i];
    return NULL;
}

static void begin_serialized_mutation(const struct config *cfg,
                                      struct proc_state *states,
                                      struct proc_state *state,
                                      struct trace_stats *stats,
                                      pid_t *mutation_owner,
                                      struct group_mutation *group) {
    if (*mutation_owner != 0 || !state || !state->active ||
        !state->stopped || !state->waiting_mutation) {
        errno = EPROTO;
        die("serialized mutation state drifted");
    }
    quiesce_tracees(cfg, states, state, stats);
    start_group_destructive_mutation(states, state, group);
    state->waiting_mutation = 0;
    state->in_mutation = 1;
    *mutation_owner = state->pid;
    resume_tracee(state, PTRACE_SYSCALL, 0,
                  "enter serialized mutation syscall");
}

static void resume_waiting_mutation(const struct config *cfg,
                                    struct proc_state *states,
                                    struct trace_stats *stats,
                                    pid_t *mutation_owner,
                                    struct group_mutation *group) {
    if (*mutation_owner != 0 || group->kind != GROUP_MUTATION_NONE) return;
    struct proc_state *next = next_waiting_mutation(states);
    if (next) {
        begin_serialized_mutation(
            cfg, states, next, stats, mutation_owner, group
        );
        return;
    }
    resume_quiesced_tracees(states);
}

static int finish_group_destructive_mutation(
        struct proc_state *states, struct group_mutation *group,
        pid_t *mutation_owner) {
    if (!group_destructive_drain_complete(states, group)) return 0;
    clear_group_destructive_mutation(states, group);
    *mutation_owner = 0;
    return 1;
}

static void complete_exec_mutation(const struct config *cfg,
                                   struct proc_state *states,
                                   struct proc_state *state,
                                   struct trace_stats *stats,
                                   pid_t *mutation_owner,
                                   struct group_mutation *group) {
    if (!state || !state->active || !state->in_mutation ||
        *mutation_owner != state->pid ||
        group->kind != GROUP_MUTATION_EXEC ||
        group->survivor_pid != state->pid) {
        errno = EPROTO;
        die("exec mutation owner drifted");
    }
    sample_writable_high_water(cfg, stats);
    state->in_mutation = 0;
    state->group_exit_expected = 0;
    state->group_exit_stop_seen = 0;
    finish_group_destructive_mutation(states, group, mutation_owner);
}

static void require_failed_exec_syscall(pid_t pid) {
    struct user_regs_struct registers;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &registers) != 0)
        die("read failed exec registers");
    long long result = (long long)registers.rax;
    if (result >= 0 || result < -4095LL) {
        errno = EPROTO;
        die("exec syscall returned without exec event");
    }
}

static void classify_background_after_main_exit(
        struct proc_state *states, struct trace_stats *stats,
        int main_exited, const struct group_mutation *group) {
    if (!main_exited || stats->active <= 0 || stats->security_violation ||
        stats->background_process || group->kind != GROUP_MUTATION_NONE)
        return;
    stats->background_process = 1;
    kill_tracees(states);
}

static void trace_sandbox(const struct config *cfg, pid_t child) {
    struct proc_state *states = calloc(MAX_TRACKED_PROCS, sizeof(*states));
    if (!states) die("allocate trace process table");
    struct trace_stats stats;
    memset(&stats, 0, sizeof(stats));
    stats.exit_code = 125;
    sample_writable_high_water(cfg, &stats);
    struct proc_state *initial = add_state(states, child, &stats);
    initial->stopped = 1;
    resume_tracee(initial, PTRACE_CONT, 0, "continue seized child");
    int status;
    int main_exited = 0;
    pid_t mutation_owner = 0;
    struct group_mutation group;
    memset(&group, 0, sizeof(group));
    while (stats.active > 0) {
        int wait_options = __WALL;
        if (group.kind != GROUP_MUTATION_NONE) {
            if (monotonic_milliseconds() >= group.deadline_ms) {
                errno = ETIMEDOUT;
                die("group-destructive mutation drain timed out");
            }
            wait_options |= WNOHANG;
        }
        pid_t pid = waitpid(-1, &status, wait_options);
        if (pid == 0) {
            struct timespec pause = {.tv_sec = 0, .tv_nsec = 1000000L};
            while (nanosleep(&pause, &pause) != 0 && errno == EINTR) {}
            continue;
        }
        if (pid < 0) {
            if (errno == EINTR) continue;
            die("wait tracee");
        }
        if (WIFEXITED(status) || WIFSIGNALED(status)) {
            struct proc_state *state = find_state(states, pid);
            if (!state) {
                errno = EPROTO;
                die("untracked ptrace process exit");
            }
            int expected_group_exit =
                group.kind != GROUP_MUTATION_NONE &&
                state->group_exit_expected && state->tgid == group.tgid;
            if (group.kind != GROUP_MUTATION_NONE && !expected_group_exit) {
                errno = EBUSY;
                die("unrelated exit during group-destructive mutation");
            }
            if (expected_group_exit && !state->group_exit_stop_seen) {
                errno = EPROTO;
                die("group member exited without exit event stop");
            }
            if (group.kind == GROUP_MUTATION_EXEC &&
                mutation_owner == pid) {
                errno = EPROTO;
                die("exec mutation owner exited before exec event");
            }
            sample_writable_high_water(cfg, &stats);
            collect_io(state, &stats);
            if (WIFSIGNALED(status) && WTERMSIG(status) == SIGSYS)
                stats.security_violation = 1;
            if (WIFSIGNALED(status) && WTERMSIG(status) == SIGXFSZ)
                stats.file_limit = 1;
            state->active = 0;
            state->stopped = 0;
            state->waiting_mutation = 0;
            state->in_mutation = 0;
            state->group_exit_expected = 0;
            state->group_exit_stop_seen = 0;
            stats.active--;
            if (pid == child) {
                main_exited = 1;
                stats.exit_code = WIFEXITED(status)
                    ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
            }
            if (group.kind != GROUP_MUTATION_NONE)
                finish_group_destructive_mutation(
                    states, &group, &mutation_owner
                );
            else if (mutation_owner == pid)
                mutation_owner = 0;
            classify_background_after_main_exit(
                states, &stats, main_exited, &group
            );
            resume_waiting_mutation(
                cfg, states, &stats, &mutation_owner, &group
            );
            continue;
        }
        if (!WIFSTOPPED(status)) {
            errno = EPROTO;
            die("unexpected tracee wait status");
        }
        int signal_number = WSTOPSIG(status);
        unsigned int event = (unsigned int)status >> 16;
        struct proc_state *state;
        if (event == PTRACE_EVENT_EXEC &&
            group.kind == GROUP_MUTATION_EXEC) {
            state = remap_exec_owner(
                states, pid, &stats, &group, &mutation_owner
            );
        } else {
            state = find_state(states, pid);
        }
        if (!state) {
            errno = EPROTO;
            die("untracked ptrace process stop");
        }
        if (event == PTRACE_EVENT_EXEC && signal_number != SIGTRAP) {
            errno = EPROTO;
            die("exec event signal drifted");
        }
        if (event == PTRACE_EVENT_EXIT &&
            group.kind != GROUP_MUTATION_NONE &&
            state->group_exit_expected && state->tgid == group.tgid) {
            if (signal_number != SIGTRAP) {
                errno = EPROTO;
                die("group exit event signal drifted");
            }
            if (!state->stopped) state->stopped = 1;
            if (!handle_group_exit_stop(cfg, state, &stats, &group)) {
                errno = EPROTO;
                die("group exit stop was not handled");
            }
            continue;
        }
        if (state->stopped) {
            errno = EPROTO;
            die("duplicate stopped tracee status");
        }
        state->stopped = 1;
        if (signal_number == SIGSYS) {
            stats.security_violation = 1;
            clear_group_destructive_mutation(states, &group);
            mutation_owner = 0;
            kill_tracees(states);
            continue;
        }
        if (event == PTRACE_EVENT_SECCOMP) {
            if (group.kind != GROUP_MUTATION_NONE ||
                state->waiting_mutation || state->in_mutation) {
                errno = EPROTO;
                die("duplicate mutation trace stop");
            }
            state->waiting_mutation = 1;
            resume_waiting_mutation(
                cfg, states, &stats, &mutation_owner, &group
            );
            continue;
        }
        if (state->in_mutation &&
            event != PTRACE_EVENT_EXEC && event != PTRACE_EVENT_EXIT &&
            signal_number != (SIGTRAP | 0x80)) {
            errno = EPROTO;
            die("unexpected stop during serialized mutation");
        }
        if (event == PTRACE_EVENT_EXEC) {
            if (!state->in_mutation || group.kind != GROUP_MUTATION_EXEC) {
                errno = EPROTO;
                die("unexpected exec event");
            }
            complete_exec_mutation(
                cfg, states, state, &stats, &mutation_owner, &group
            );
            resume_waiting_mutation(
                cfg, states, &stats, &mutation_owner, &group
            );
            continue;
        }
        if (signal_number == (SIGTRAP | 0x80) && state->in_mutation) {
            if (mutation_owner != pid) {
                errno = EPROTO;
                die("mutation syscall exit owner drifted");
            }
            sample_writable_high_water(cfg, &stats);
            state->in_mutation = 0;
            if (group.kind == GROUP_MUTATION_EXEC) {
                require_failed_exec_syscall(pid);
                clear_group_destructive_mutation(states, &group);
            } else if (group.kind != GROUP_MUTATION_NONE) {
                errno = EPROTO;
                die("group-destructive syscall returned unexpectedly");
            }
            mutation_owner = 0;
            resume_waiting_mutation(
                cfg, states, &stats, &mutation_owner, &group
            );
            continue;
        }
        if (group.kind != GROUP_MUTATION_NONE) {
            errno = EBUSY;
            die("unrelated stop during group-destructive mutation");
        }
        if (event == PTRACE_EVENT_FORK || event == PTRACE_EVENT_VFORK ||
            event == PTRACE_EVENT_CLONE) {
            unsigned long new_pid = 0;
            if (ptrace(PTRACE_GETEVENTMSG, pid, NULL, &new_pid) != 0 ||
                new_pid == 0) die("read forked tracee identity");
            struct proc_state *created = add_state(states, (pid_t)new_pid, &stats);
            created->awaiting_stop = 1;
        } else if (event == PTRACE_EVENT_EXIT) {
            if (!state->in_mutation && !stats.security_violation &&
                !stats.background_process) {
                if (mutation_owner != 0) {
                    errno = EBUSY;
                    die("concurrent exit during serialized mutation");
                }
                quiesce_tracees(cfg, states, state, &stats);
            }
            sample_writable_high_water(cfg, &stats);
            collect_io(state, &stats);
        }
        int deliver = (signal_number == SIGTRAP || signal_number == SIGSTOP ||
                       signal_number == (SIGTRAP | 0x80))
            ? 0 : signal_number;
        resume_tracee(state, PTRACE_CONT, deliver, "continue tracee");
    }
    if (!main_exited && stats.exit_code == 125) stats.exit_code = 137;
    sample_writable_high_water(cfg, &stats);
    write_stats(cfg, &stats);
    free(states);
}

static int namespace_main(struct config *cfg) {
    if (mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0)
        die("make mounts private");
    char merged[PATH_MAX];
    setup_mounts(cfg, merged, sizeof(merged));
    close(cfg->rootfs_fd);
    for (size_t i = 0; i < cfg->device_count; i++) close(cfg->devices[i].fd);
    close(cfg->public_fd);
    close(cfg->workspace_fd);
    close(cfg->submission_fd);
    if (cfg->memory_fd >= 0) close(cfg->memory_fd);
    close(cfg->tmp_fd);
    close(cfg->shm_fd);
    enter_root(merged);
    snprintf(merged, sizeof(merged), "/");

    int ready_pipe[2], go_pipe[2];
    if (pipe2(ready_pipe, O_CLOEXEC) != 0 || pipe2(go_pipe, O_CLOEXEC) != 0)
        die("create user-map pipes");
    if (fcntl(cfg->stats_fd, F_SETFD, FD_CLOEXEC) != 0) die("protect stats fd");
    pid_t child = fork();
    if (child < 0) die("fork sandbox child");
    if (child == 0) {
        close(ready_pipe[0]);
        close(go_pipe[1]);
        sandbox_child(cfg, ready_pipe[1], go_pipe[0]);
    }
    close(ready_pipe[1]);
    close(go_pipe[0]);
    char ready;
    if (read(ready_pipe[0], &ready, 1) != 1) die("await user namespace");
    install_user_map(merged, child, cfg->host_uid, cfg->host_gid);
    long trace_options = PTRACE_O_TRACEEXEC | PTRACE_O_TRACEFORK |
                         PTRACE_O_TRACEVFORK | PTRACE_O_TRACECLONE |
                         PTRACE_O_TRACEEXIT | PTRACE_O_TRACESECCOMP |
                         PTRACE_O_TRACESYSGOOD | PTRACE_O_EXITKILL;
    if (ptrace(PTRACE_SEIZE, child, NULL, (void *)trace_options) != 0)
        die("seize sandbox child");
    if (ptrace(PTRACE_INTERRUPT, child, NULL, NULL) != 0)
        die("interrupt seized sandbox child");
    int initial_status;
    if (waitpid(child, &initial_status, __WALL) != child ||
        !WIFSTOPPED(initial_status) ||
        ((unsigned int)initial_status >> 16) != PTRACE_EVENT_STOP)
        die("wait seized sandbox child");
    write_all(go_pipe[1], "G");
    close(ready_pipe[0]);
    close(go_pipe[1]);
    trace_sandbox(cfg, child);
    close(cfg->quota_fd);
    close(cfg->stats_fd);
    return 0;
}

static void await_cgroup_admission(struct config *cfg) {
    char token;
    ssize_t size;
    do {
        size = read(cfg->start_fd, &token, 1);
    } while (size < 0 && errno == EINTR);
    if (size != 1 || token != 'G') {
        errno = EPROTO;
        die("await cgroup admission");
    }
    close(cfg->start_fd);
    cfg->start_fd = -1;
}

int main(int argc, char **argv) {
    pid_t runner_pid = getppid();
    arm_parent_death_signal(runner_pid);
    struct config cfg;
    parse_config(argc, argv, &cfg);
    await_cgroup_admission(&cfg);
    int namespace_flags = CLONE_NEWNS | CLONE_NEWIPC | CLONE_NEWUTS |
                          CLONE_NEWNET | CLONE_NEWPID | CLONE_NEWCGROUP;
    if (unshare(namespace_flags) != 0) die("create isolation namespaces");
    pid_t supervisor_pid = getpid();
    pid_t child = fork();
    if (child < 0) die("fork PID namespace supervisor");
    if (child == 0) {
        arm_parent_death_signal(supervisor_pid);
        _exit(namespace_main(&cfg));
    }
    int status;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        die("wait PID namespace supervisor");
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    errno = ECHILD;
    die("invalid PID namespace supervisor status");
}
