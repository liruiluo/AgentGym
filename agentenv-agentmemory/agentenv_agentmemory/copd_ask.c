/* Ordinary sandbox CLI. Authentication and provider traffic stay in the broker. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#define LIMIT 32768

static int invalid(void) {
    puts("[teacher_error:invalid_question]");
    return 2;
}

static int write_all(int fd, const char *data, size_t size) {
    while (size) {
        ssize_t n = write(fd, data, size);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        data += n;
        size -= (size_t)n;
    }
    return 0;
}

int main(int argc, char **argv) {
    char question[LIMIT + 2];
    size_t size = 0;
    if (argc == 2 && strcmp(argv[1], "--stdin") == 0) {
        while (size <= LIMIT) {
            ssize_t n = read(STDIN_FILENO, question + size, LIMIT + 1 - size);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) return invalid();
            if (n == 0) break;
            size += (size_t)n;
        }
        if (memchr(question, '\0', size)) return invalid();
    } else {
        for (int i = 1; i < argc; i++) {
            size_t n = strlen(argv[i]);
            size_t separator = i > 1 ? 1 : 0;
            if (n + size + separator > LIMIT) return invalid();
            if (separator) question[size++] = ' ';
            memcpy(question + size, argv[i], n);
            size += n;
        }
    }
    if (!size || size > LIMIT) return invalid();
    question[size++] = '\0';
    int lock = open("/run/copd/lock", O_RDONLY | O_CLOEXEC);
    if (lock < 0 || flock(lock, LOCK_EX) != 0) return 3;
    int request = open("/run/copd/request", O_WRONLY | O_CLOEXEC);
    if (request < 0 || write_all(request, question, size) != 0) return 3;
    close(request);
    int reply = open("/run/copd/reply", O_RDONLY | O_CLOEXEC);
    if (reply < 0) return 3;
    char buffer[4096];
    for (;;) {
        ssize_t n = read(reply, buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return 3;
        if (n == 0) break;
        if (write_all(STDOUT_FILENO, buffer, (size_t)n) != 0) return 3;
    }
    close(reply);
    close(lock);
    return 0;
}
