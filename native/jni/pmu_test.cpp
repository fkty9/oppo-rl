#include <iostream>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <cerrno>
#include <vector>
#include <sys/syscall.h>
// #define __NR_perf_event_open 241

// 定义打开PMU事件的函数
static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                           int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int main() {
    struct Event {
        const char* name;
        struct perf_event_attr pe;
        int fd;
        long long count;
    };

    // 定义需要捕获的事件
    std::vector<Event> events = {
        {"CPU cycles", {0}, -1, 0},
        {"Instructions", {0}, -1, 0},
        {"L1 cache misses", {0}, -1, 0},
        {"L1 cache accesses", {0}, -1, 0},
        {"Branch misses", {0}, -1, 0},
        {"Bus cycles", {0}, -1, 0},
    };

    // 初始化事件属性
    for (auto &event : events) {
        memset(&event.pe, 0, sizeof(struct perf_event_attr));
        event.pe.size = sizeof(struct perf_event_attr);
        event.pe.disabled = 1;
        event.pe.exclude_kernel = 1;
        event.pe.exclude_hv = 1;
    }

    events[0].pe.type = PERF_TYPE_HARDWARE;
    events[0].pe.config = PERF_COUNT_HW_CPU_CYCLES;

    events[1].pe.type = PERF_TYPE_HARDWARE;
    events[1].pe.config = PERF_COUNT_HW_INSTRUCTIONS;

    events[2].pe.type = PERF_TYPE_HW_CACHE;
    events[2].pe.config = (PERF_COUNT_HW_CACHE_L1D | 
                           (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                           (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));

    events[3].pe.type = PERF_TYPE_HW_CACHE;
    events[3].pe.config = (PERF_COUNT_HW_CACHE_L1D | 
                           (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                           (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));


    events[4].pe.type = PERF_TYPE_HARDWARE;
    events[4].pe.config = PERF_COUNT_HW_BRANCH_MISSES;

    events[5].pe.type = PERF_TYPE_HARDWARE;
    events[5].pe.config = PERF_COUNT_HW_BUS_CYCLES;


    // 打开PMU事件
    for (auto &event : events) {
        event.fd = perf_event_open(&event.pe, 0, -1, -1, 0);
        if (event.fd == -1) {
            std::cerr << "Error opening " << event.name << ": " << strerror(errno) << std::endl;
            return -1;
        }
    }

    // 启动计数
    for (auto &event : events) {
        ioctl(event.fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(event.fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    // 放置待测代码段，例如一个简单的循环
    for (volatile int i = 0; i < 1000000; ++i);

    // 停止计数
    for (auto &event : events) {
        ioctl(event.fd, PERF_EVENT_IOC_DISABLE, 0);
    }

    // 读取计数器的值
    for (auto &event : events) {
        if (read(event.fd, &event.count, sizeof(long long)) == -1) {
            std::cerr << "Error reading " << event.name << ": " << strerror(errno) << std::endl;
            return -1;
        }
    }

    // 输出结果
    for (const auto &event : events) {
        std::cout << event.name << ": " << event.count << std::endl;
    }

    // 关闭文件描述符
    for (auto &event : events) {
        close(event.fd);
    }

    return 0;
}
