#define _GNU_SOURCE

#include <link.h>
#include <stddef.h>
#include <stdint.h>

#ifndef MLE_ALLOWED_ROOTFS_PREFIX
#error "MLE_ALLOWED_ROOTFS_PREFIX must be supplied by the bundle builder"
#endif
#ifndef MLE_ALLOWED_ROOTFS_FD_PREFIX
#error "MLE_ALLOWED_ROOTFS_FD_PREFIX must be supplied by the bundle builder"
#endif

#if !defined(__x86_64__)
#error "The runtime audit module is restricted to Linux/x86_64"
#endif

#define MLE_AUDIT_STATE_FD 198
#define MLE_AUDIT_PROBE_FD 199
#define SYS_WRITE 1
#define SYS_EXIT_GROUP 231

static const char MleBridgeRuntimeAuditIdentity[] __attribute__((used)) =
    "mlebench_lite_runtime_audit_v1";
static const char allowed_rootfs_prefix[] = MLE_ALLOWED_ROOTFS_PREFIX;
static const char allowed_rootfs_fd_prefix[] = MLE_ALLOWED_ROOTFS_FD_PREFIX;
static const char audit_marker[] = "mlebench_lite_runtime_audit_v1\n";

static long raw_syscall3(long number, long first, long second, long third) {
    long result;
    __asm__ volatile(
        "syscall"
        : "=a"(result)
        : "a"(number), "D"(first), "S"(second), "d"(third)
        : "rcx", "r11", "memory");
    return result;
}

static void __attribute__((noreturn)) fail_closed(void) {
    (void)raw_syscall3(SYS_EXIT_GROUP, 126, 0, 0);
    __builtin_unreachable();
}

static int strings_equal(const char *left, const char *right) {
    size_t index = 0;
    while (left[index] != '\0' && left[index] == right[index]) index++;
    return left[index] == '\0' && right[index] == '\0';
}

static int path_has_safe_suffix(const char *path, const char *prefix) {
    size_t index = 0;
    while (prefix[index] != '\0') {
        if (path[index] != prefix[index]) return 0;
        index++;
    }
    if (path[index] != '/') return 0;
    index++;
    for (;;) {
        size_t component = index;
        while (path[index] != '\0' && path[index] != '/') index++;
        size_t length = index - component;
        if (length == 0 ||
            (length == 1 && path[component] == '.') ||
            (length == 2 && path[component] == '.' &&
             path[component + 1] == '.'))
            return 0;
        if (path[index] == '\0') return 1;
        index++;
    }
}

static int path_is_below_rootfs(const char *path) {
    return path_has_safe_suffix(path, allowed_rootfs_prefix);
}

static int path_is_below_rootfs_fd(const char *path) {
    return path_has_safe_suffix(path, allowed_rootfs_fd_prefix);
}

#ifdef MLE_RUNTIME_AUDIT_POLICY_TEST
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    if (strings_equal(argv[1], "rootfs"))
        return path_is_below_rootfs(argv[2]) ? 0 : 1;
    if (strings_equal(argv[1], "fd"))
        return path_is_below_rootfs_fd(argv[2]) ? 0 : 1;
    return 2;
}
#endif

unsigned int la_version(unsigned int version) {
    if (version < LAV_CURRENT) fail_closed();
    long state_result = raw_syscall3(
        SYS_WRITE, MLE_AUDIT_STATE_FD, (long)(uintptr_t)audit_marker,
        (long)(sizeof(audit_marker) - 1));
    long probe_result = raw_syscall3(
        SYS_WRITE, MLE_AUDIT_PROBE_FD, (long)(uintptr_t)audit_marker,
        (long)(sizeof(audit_marker) - 1));
    if (state_result != (long)(sizeof(audit_marker) - 1) &&
        probe_result != (long)(sizeof(audit_marker) - 1))
        fail_closed();
    return LAV_CURRENT;
}

unsigned int la_objopen(struct link_map *map, Lmid_t namespace_id,
                        uintptr_t *cookie) {
    (void)namespace_id;
    (void)cookie;
    if (map == NULL || map->l_name == NULL) fail_closed();
    const char *path = map->l_name;
    if (path[0] == '\0' || strings_equal(path, "linux-vdso.so.1") ||
        path_is_below_rootfs(path) || path_is_below_rootfs_fd(path))
        return LA_FLG_BINDTO | LA_FLG_BINDFROM;
    fail_closed();
}
