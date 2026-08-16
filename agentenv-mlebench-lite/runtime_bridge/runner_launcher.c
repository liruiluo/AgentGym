#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef MLE_ROOTFS_LOADER_PATH
#error "MLE_ROOTFS_LOADER_PATH must be supplied by the bundle builder"
#endif
#ifndef MLE_ROOTFS_PYTHON_PATH
#error "MLE_ROOTFS_PYTHON_PATH must be supplied by the bundle builder"
#endif
#ifndef MLE_ROOTFS_PYTHON_HOME
#error "MLE_ROOTFS_PYTHON_HOME must be supplied by the bundle builder"
#endif
#ifndef MLE_ROOTFS_LIBRARY_PATH
#error "MLE_ROOTFS_LIBRARY_PATH must be supplied by the bundle builder"
#endif
#ifndef MLE_RUNNER_SOURCE_SHA256
#error "MLE_RUNNER_SOURCE_SHA256 must be supplied by the bundle builder"
#endif
#ifndef MLE_LINUX_RUNTIME_SHA256
#error "MLE_LINUX_RUNTIME_SHA256 must be supplied by the bundle builder"
#endif
#ifndef MLE_RUNTIME_AUDIT_SHA256
#error "MLE_RUNTIME_AUDIT_SHA256 must be supplied by the bundle builder"
#endif

static const char MleBridgeLauncherIdentity[] =
    "mlebench_lite_native_runner_launcher_v1";
static const char rootfs_loader_path[] = MLE_ROOTFS_LOADER_PATH;
static const char rootfs_python_path[] = MLE_ROOTFS_PYTHON_PATH;
static const char rootfs_python_home[] = MLE_ROOTFS_PYTHON_HOME;
static const char rootfs_library_path[] = MLE_ROOTFS_LIBRARY_PATH;
static const char runtime_audit_identity[] =
    "mlebench_lite_runtime_audit_v1";
static const char runner_source_sha256[] = MLE_RUNNER_SOURCE_SHA256;
static const char linux_runtime_sha256[] = MLE_LINUX_RUNTIME_SHA256;
static const char runtime_audit_sha256[] = MLE_RUNTIME_AUDIT_SHA256;

#define MLE_BUNDLE_ROOT_FD 197
#define MLE_AUDIT_STATE_FD 198

static void __attribute__((noreturn)) fail_closed(const char *message) {
    (void)message;
    _exit(2);
}

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_length;
    unsigned char block[64];
    size_t block_length;
};

static uint32_t rotate_right(uint32_t value, unsigned int amount) {
    return (value >> amount) | (value << (32U - amount));
}

static void sha256_transform(struct sha256_context *context) {
    static const uint32_t constants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    uint32_t words[64];
    for (size_t index = 0; index < 16; index++) {
        size_t offset = index * 4;
        words[index] = ((uint32_t)context->block[offset] << 24) |
                       ((uint32_t)context->block[offset + 1] << 16) |
                       ((uint32_t)context->block[offset + 2] << 8) |
                       (uint32_t)context->block[offset + 3];
    }
    for (size_t index = 16; index < 64; index++) {
        uint32_t left = words[index - 15];
        uint32_t right = words[index - 2];
        uint32_t small0 = rotate_right(left, 7) ^ rotate_right(left, 18) ^
                          (left >> 3);
        uint32_t small1 = rotate_right(right, 17) ^ rotate_right(right, 19) ^
                          (right >> 10);
        words[index] = words[index - 16] + small0 + words[index - 7] + small1;
    }
    uint32_t a = context->state[0], b = context->state[1];
    uint32_t c = context->state[2], d = context->state[3];
    uint32_t e = context->state[4], f = context->state[5];
    uint32_t g = context->state[6], h = context->state[7];
    for (size_t index = 0; index < 64; index++) {
        uint32_t big1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                        rotate_right(e, 25);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t first = h + big1 + choice + constants[index] + words[index];
        uint32_t big0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                        rotate_right(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t second = big0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + first;
        d = c;
        c = b;
        b = a;
        a = first + second;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void sha256_init(struct sha256_context *context) {
    memset(context, 0, sizeof(*context));
    context->state[0] = 0x6a09e667U;
    context->state[1] = 0xbb67ae85U;
    context->state[2] = 0x3c6ef372U;
    context->state[3] = 0xa54ff53aU;
    context->state[4] = 0x510e527fU;
    context->state[5] = 0x9b05688cU;
    context->state[6] = 0x1f83d9abU;
    context->state[7] = 0x5be0cd19U;
}

static void sha256_update(struct sha256_context *context,
                          const unsigned char *payload, size_t length) {
    for (size_t index = 0; index < length; index++) {
        context->block[context->block_length++] = payload[index];
        if (context->block_length == sizeof(context->block)) {
            sha256_transform(context);
            context->bit_length += 512U;
            context->block_length = 0;
        }
    }
}

static void sha256_final(struct sha256_context *context,
                         unsigned char digest[32]) {
    size_t index = context->block_length;
    context->block[index++] = 0x80U;
    if (index > 56) {
        while (index < 64) context->block[index++] = 0;
        sha256_transform(context);
        index = 0;
    }
    while (index < 56) context->block[index++] = 0;
    context->bit_length += (uint64_t)context->block_length * 8U;
    for (size_t offset = 0; offset < 8; offset++)
        context->block[63 - offset] =
            (unsigned char)(context->bit_length >> (offset * 8));
    sha256_transform(context);
    for (size_t word = 0; word < 8; word++)
        for (size_t byte = 0; byte < 4; byte++)
            digest[word * 4 + byte] =
                (unsigned char)(context->state[word] >> (24 - byte * 8));
}

static int valid_sha256(const char *value) {
    if (!value || strlen(value) != 64) return 0;
    for (size_t index = 0; index < 64; index++)
        if (!((value[index] >= '0' && value[index] <= '9') ||
              (value[index] >= 'a' && value[index] <= 'f')))
            return 0;
    return 1;
}

static void require_fd_sha256(int descriptor, const char *expected) {
    if (!valid_sha256(expected)) fail_closed("invalid expected SHA256");
    struct stat metadata;
    if (fstat(descriptor, &metadata) != 0 || metadata.st_size < 0 ||
        metadata.st_size > 2 * 1024 * 1024)
        fail_closed("trusted member size drifted");
    struct sha256_context context;
    sha256_init(&context);
    unsigned char buffer[16384];
    off_t offset = 0;
    for (;;) {
        ssize_t length = pread(descriptor, buffer, sizeof(buffer), offset);
        if (length < 0) {
            if (errno == EINTR) continue;
            fail_closed("trusted member hash read failed");
        }
        if (length == 0) break;
        sha256_update(&context, buffer, (size_t)length);
        offset += length;
    }
    unsigned char digest[32];
    char hex[65];
    static const char alphabet[] = "0123456789abcdef";
    sha256_final(&context, digest);
    for (size_t index = 0; index < sizeof(digest); index++) {
        hex[index * 2] = alphabet[digest[index] >> 4];
        hex[index * 2 + 1] = alphabet[digest[index] & 15U];
    }
    hex[64] = '\0';
    if (strcmp(hex, expected) != 0)
        fail_closed("trusted member SHA256 drifted");
}

static void require_sealed_regular(int descriptor, uid_t owner, int executable);

static void verify_artifact_lock_digest(int directory,
                                        const char *expected_runtime_digest) {
    int descriptor = openat(directory, "artifact-lock.json",
                            O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0) fail_closed("artifact lock is unavailable");
    require_sealed_regular(descriptor, geteuid(), 0);
    require_fd_sha256(descriptor, expected_runtime_digest);
    close(descriptor);
}

static void arm_parent_death_signal(void) {
    pid_t parent = getppid();
    if (parent <= 1) fail_closed("launcher parent is absent");
    if (prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0) != 0)
        fail_closed("PR_SET_PDEATHSIG failed");
    if (getppid() != parent) fail_closed("launcher parent changed");
}

static void close_untrusted_descriptors(void) {
#ifdef __NR_close_range
    if (syscall(__NR_close_range, 3U, ~0U, 0U) == 0) return;
    if (errno != ENOSYS && errno != EINVAL)
        fail_closed("close inherited descriptors failed");
#endif
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0)
        fail_closed("descriptor limit is unavailable");
    unsigned long maximum =
        limit.rlim_cur == RLIM_INFINITY ? 1048576UL : (unsigned long)limit.rlim_cur;
    for (unsigned long descriptor = 3; descriptor < maximum; descriptor++)
        close((int)descriptor);
}

static int safe_component(const char *value) {
    return value[0] != '\0' && strcmp(value, ".") != 0 &&
           strcmp(value, "..") != 0 && strchr(value, '/') == NULL;
}

static int open_absolute_nofollow(const char *path, int final_flags) {
    if (!path || path[0] != '/' || path[1] == '\0' || strlen(path) >= PATH_MAX)
        fail_closed("unsafe trusted path");
    char copy[PATH_MAX];
    if (snprintf(copy, sizeof(copy), "%s", path) >= (int)sizeof(copy))
        fail_closed("trusted path is too long");
    int current = open("/", O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (current < 0) fail_closed("cannot open trusted root");
    char *save = NULL;
    char *component = strtok_r(copy + 1, "/", &save);
    while (component) {
        if (!safe_component(component)) fail_closed("unsafe path component");
        char *next = strtok_r(NULL, "/", &save);
        int flags = next ? (O_PATH | O_DIRECTORY) : final_flags;
        int opened = openat(current, component, flags | O_NOFOLLOW | O_CLOEXEC);
        close(current);
        if (opened < 0) fail_closed("trusted path cannot be opened");
        current = opened;
        component = next;
    }
    return current;
}

static void require_regular_identity(int descriptor, uid_t owner, int executable,
                                     mode_t forbidden_write_bits) {
    struct stat value;
    if (fstat(descriptor, &value) != 0 || !S_ISREG(value.st_mode) ||
        value.st_uid != owner || value.st_nlink != 1 ||
        (value.st_mode & forbidden_write_bits) != 0 ||
        (executable && (value.st_mode & S_IXUSR) == 0))
        fail_closed("sealed regular-file identity drifted");
}

static void require_sealed_regular(int descriptor, uid_t owner, int executable) {
    struct stat value;
    if (fstat(descriptor, &value) != 0 ||
        (value.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0)
        fail_closed("sealed regular-file mode drifted");
    require_regular_identity(descriptor, owner, executable,
                             S_IWUSR | S_IWGRP | S_IWOTH);
}

static void require_read_only_mount_regular(int descriptor, uid_t owner,
                                            int executable) {
    require_regular_identity(descriptor, owner, executable,
                             S_IWGRP | S_IWOTH);
}

static void require_read_only_mount(const char *path) {
    struct statvfs value;
    if (statvfs(path, &value) != 0 || (value.f_flag & ST_RDONLY) == 0)
        fail_closed("rootfs is not a read-only mount");
}

static int open_bundle_anchor(const char *expected_runtime_digest,
                              int *runner_descriptor) {
    char launcher[PATH_MAX];
    ssize_t length = readlink("/proc/self/exe", launcher, sizeof(launcher) - 1);
    if (length <= 0 || length >= (ssize_t)sizeof(launcher) - 1)
        fail_closed("launcher identity is unavailable");
    launcher[length] = '\0';
    char *slash = strrchr(launcher, '/');
    if (!slash || strcmp(slash + 1, "sandbox-runner") != 0)
        fail_closed("launcher name drifted");
    *slash = '\0';
    int directory = open_absolute_nofollow(launcher, O_RDONLY | O_DIRECTORY);
    struct stat directory_metadata;
    if (fstat(directory, &directory_metadata) != 0 ||
        !S_ISDIR(directory_metadata.st_mode) ||
        directory_metadata.st_uid != geteuid() ||
        (directory_metadata.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0)
        fail_closed("bundle directory identity drifted");
    int candidate = openat(directory, "sandbox-runner",
                           O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    int running = open("/proc/self/exe", O_PATH | O_CLOEXEC);
    struct stat candidate_metadata, running_metadata;
    if (candidate < 0 || running < 0 ||
        fstat(candidate, &candidate_metadata) != 0 ||
        fstat(running, &running_metadata) != 0 ||
        candidate_metadata.st_dev != running_metadata.st_dev ||
        candidate_metadata.st_ino != running_metadata.st_ino)
        fail_closed("bundle launcher does not match the running executable");
    close(running);
    require_sealed_regular(candidate, geteuid(), 1);
    verify_artifact_lock_digest(directory, expected_runtime_digest);
    int source = openat(directory, "runner.py", O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (source < 0) fail_closed("bundle runner source is unavailable");
    require_sealed_regular(source, geteuid(), 0);
    require_fd_sha256(source, runner_source_sha256);
    int runtime = openat(directory, "linux_runtime.py",
                         O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (runtime < 0) fail_closed("bundle Linux runtime is unavailable");
    require_sealed_regular(runtime, geteuid(), 0);
    require_fd_sha256(runtime, linux_runtime_sha256);
    close(runtime);
    close(candidate);
    if (directory != MLE_BUNDLE_ROOT_FD) {
        if (dup2(directory, MLE_BUNDLE_ROOT_FD) != MLE_BUNDLE_ROOT_FD)
            fail_closed("bundle anchor descriptor cannot be reserved");
        close(directory);
    }
    if (fcntl(MLE_BUNDLE_ROOT_FD, F_SETFD, 0) != 0)
        fail_closed("bundle anchor descriptor cannot be inherited");
    *runner_descriptor = source;
    return MLE_BUNDLE_ROOT_FD;
}

static int create_audit_state(void) {
#ifdef __NR_memfd_create
    int descriptor = (int)syscall(__NR_memfd_create, "mlebridge-runtime-audit", 1U);
    if (descriptor < 0) fail_closed("runtime audit state cannot be created");
#else
    fail_closed("runtime audit state requires memfd_create");
#endif
    if (ftruncate(descriptor, 4096) != 0)
        fail_closed("runtime audit state cannot be sized");
    if (descriptor != MLE_AUDIT_STATE_FD) {
        if (dup2(descriptor, MLE_AUDIT_STATE_FD) != MLE_AUDIT_STATE_FD)
            fail_closed("runtime audit descriptor cannot be reserved");
        close(descriptor);
    }
    if (fcntl(MLE_AUDIT_STATE_FD, F_SETFD, 0) != 0)
        fail_closed("runtime audit descriptor cannot be inherited");
    return MLE_AUDIT_STATE_FD;
}

static int verify_audited_runtime_mappings(int bundle_fd) {
    int library = openat(bundle_fd, "lib",
                         O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    struct stat metadata;
    if (library < 0 || fstat(library, &metadata) != 0 ||
        !S_ISDIR(metadata.st_mode) || metadata.st_uid != geteuid() ||
        (metadata.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0)
        fail_closed("runtime audit directory drifted");
    int descriptor = openat(library, "mlebench-lite-runtime-audit.so",
                            O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    close(library);
    if (descriptor < 0) fail_closed("runtime audit module is unavailable");
    require_sealed_regular(descriptor, geteuid(), 0);
    require_fd_sha256(descriptor, runtime_audit_sha256);
    if (fcntl(descriptor, F_SETFD, 0) != 0)
        fail_closed("runtime audit module cannot be inherited");
    return descriptor;
}

int main(int argc, char **argv) {
    (void)MleBridgeLauncherIdentity;
    if (argc != 4 || strcmp(argv[1], "--expected-runtime-digest") != 0 ||
        !valid_sha256(argv[2]) ||
        (strcmp(argv[3], "attest") != 0 && strcmp(argv[3], "execute") != 0 &&
         strcmp(argv[3], "freeze") != 0 && strcmp(argv[3], "teardown") != 0))
        fail_closed("launcher operation drifted");
    arm_parent_death_signal();
    close_untrusted_descriptors();
    require_read_only_mount(rootfs_python_home);

    int runner_fd = -1;
    int bundle_fd = open_bundle_anchor(argv[2], &runner_fd);
    int loader_fd = open_absolute_nofollow(rootfs_loader_path, O_RDONLY);
    int python_fd = open_absolute_nofollow(rootfs_python_path, O_RDONLY);
    int audit_fd = verify_audited_runtime_mappings(bundle_fd);
    int audit_state_fd = create_audit_state();
    require_read_only_mount_regular(loader_fd, 0, 1);
    require_read_only_mount_regular(python_fd, 0, 1);
    require_sealed_regular(runner_fd, geteuid(), 0);
    if (fcntl(python_fd, F_SETFD, 0) != 0 ||
        fcntl(runner_fd, F_SETFD, 0) != 0)
        fail_closed("trusted descriptor inheritance failed");

    char bundle_root[64], python_descriptor[64], runner_descriptor[64],
        audit_descriptor[64];
    if (snprintf(bundle_root, sizeof(bundle_root), "/proc/self/fd/%d",
                 bundle_fd) >= (int)sizeof(bundle_root) ||
        snprintf(python_descriptor, sizeof(python_descriptor),
                 "/proc/self/fd/%d", python_fd) >=
            (int)sizeof(python_descriptor) ||
        snprintf(runner_descriptor, sizeof(runner_descriptor),
                 "/proc/self/fd/%d", runner_fd) >=
            (int)sizeof(runner_descriptor) ||
        snprintf(audit_descriptor, sizeof(audit_descriptor),
                 "/proc/self/fd/%d", audit_fd) >=
            (int)sizeof(audit_descriptor))
        fail_closed("trusted descriptor path is too long");

    if (clearenv() != 0 || setenv("PYTHONHOME", rootfs_python_home, 1) != 0 ||
        setenv("PYTHONDONTWRITEBYTECODE", "1", 1) != 0 ||
        setenv("MLE_BRIDGE_BUNDLE_ROOT", bundle_root, 1) != 0 ||
        setenv("MLE_BRIDGE_EXPECTED_RUNTIME_DIGEST", argv[2], 1) != 0 ||
        setenv("MLE_BRIDGE_RUNTIME_AUDIT", runtime_audit_identity, 1) != 0 ||
        setenv("PATH", "/usr/bin:/bin", 1) != 0 ||
        setenv("LANG", "C", 1) != 0 || setenv("LC_ALL", "C", 1) != 0)
        fail_closed("launcher environment setup failed");

    static const char bootstrap[] =
        "import runpy,sys;"
        "b=sys.argv.pop(1);p=sys.argv.pop(1);"
        "sys.path.insert(0,b);runpy.run_path(p,run_name='__main__')";
    char *const loader_argv[] = {
        (char *)rootfs_loader_path,
        "--inhibit-cache",
        "--audit",
        audit_descriptor,
        "--library-path",
        (char *)rootfs_library_path,
        python_descriptor,
        "-P",
        "-S",
        "-s",
        "-B",
        "-c",
        (char *)bootstrap,
        bundle_root,
        runner_descriptor,
        argv[3],
        NULL,
    };
    extern char **environ;
    (void)audit_state_fd;
    fexecve(loader_fd, loader_argv, environ);
    fail_closed("sealed rootfs Python launch failed");
}
