#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include <vector>
#include "fps.h"
#include "state.h"
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <cerrno>
#include <cstdint>
#define PORT 8888

void red() {
    printf("\033[1;31m");
}
void yellow() {
    printf("\033[1;33m");
}
void reset() {
    printf("\033[0m");
}

typedef struct {
    int user;
    int nice;
    int system;
    int idle;
    int iowait;
    int irq;
    int softirq;
} CPUData;

typedef struct {
    CPUData initial[MAX_CPU_COUNT];
    CPUData current[MAX_CPU_COUNT];
} CPUControl;

void parse_cpu_data(const char* line, CPUData* data) {
    sscanf(line, "cpu %*d %d %d %d %d %d %d %d", &data->user, &data->nice, &data->system, &data->idle, &data->iowait, &data->irq, &data->softirq);
}

void initialize_cpu_control(CPUControl* control) {
    // system("adb shell cat /proc/stat > stat.txt");
    FILE* file = fopen("/proc/stat", "r");
    if (file == NULL) {
        printf("Failed to open stat.txt\n");
        return;
    }
    char line[MAX_LINE_LENGTH];
    fgets(line, MAX_LINE_LENGTH, file);  // Skip the first line

    for (int i = 0; i < MAX_CPU_COUNT; i++) {
        fgets(line, MAX_LINE_LENGTH, file);
        parse_cpu_data(line, &control->initial[i]);
    }
    fclose(file);
}

void update_cpu_utilization(CPUControl* control, double* utilization) {
    // system("adb shell cat /proc/stat > stat2.txt");
    usleep(10000);
    FILE* file = fopen("/proc/stat", "r");
    if (file == NULL) {
        printf("Failed to open stat.txt\n");
        return;
    }

    char line[MAX_LINE_LENGTH];
    fgets(line, MAX_LINE_LENGTH, file);  // Skip the first line

    for (int i = 0; i < MAX_CPU_COUNT; i++) {
        fgets(line, MAX_LINE_LENGTH, file);
        parse_cpu_data(line, &control->current[i]);

        CPUData* initial = &control->initial[i];
        CPUData* current = &control->current[i];

        int curr_time = current->user + current->nice + current->system + current->idle + current->iowait + current->irq + current->softirq;
        int initial_time = initial->user + initial->nice + initial->system + initial->idle + initial->iowait + initial->irq + initial->softirq;
        int interval = curr_time - initial_time;

        double cpu_util = ((current->user + current->system + current->nice) - (initial->user + initial->system + initial->nice)) / (double)interval;
        utilization[i] = cpu_util;

        // Update initial data
        memcpy(initial, current, sizeof(CPUData));
    }

    fclose(file);
}

int get_gpu_freq() {
    const char* filename = "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq";
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return -1;
    }

    int freq;
    if (fscanf(file, "%d", &freq) != 1) {
        printf("Failed to read frequency from file: %s\n", filename);
        fclose(file);
        return -1;
    }
    fclose(file);
    return freq;
}

float get_gpu_util() {
    const char* filename = "/sys/class/kgsl/kgsl-3d0/gpubusy";
    FILE* fp = fopen(filename, "r");
    int a, b;
    fscanf(fp, "%d%d", &a, &b);
    // printf("here in gpu_util %d, %d", a, b);
    if (a == 0 || b == 0) {
        return 0.0;
    }
    return (float)a / b;
}

// int get_sbig_cpu_freq() {
//     const char* filename = "/sys/devices/system/cpu/cpufreq/policy7/scaling_cur_freq";
//     FILE* file = fopen(filename, "r");
//     if (file == NULL) {
//         printf("Failed to open file: %s\n", filename);
//         return -1;
//     }

//     int freq;
//     if (fscanf(file, "%d", &freq) != 1) {
//         printf("Failed to read frequency from file: %s\n", filename);
//         fclose(file);
//         return -1;
//     }

//     fclose(file);
//     return freq;
// }

int get_big_cpu_freq() {
    const char* filename = "/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq";
    printf("%s\n", filename);
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return -1;
    }

    int freq;
    if (fscanf(file, "%d", &freq) != 1) {
        printf("Failed to read frequency from file: %s\n", filename);
        fclose(file);
        return -1;
    }

    fclose(file);
    return freq;
}

int get_little_cpu_freq() {
    const char* filename = "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq";
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return -1;
    }

    int freq;
    if (fscanf(file, "%d", &freq) != 1) {
        printf("Failed to read frequency from file: %s\n", filename);
        fclose(file);
        return -1;
    }

    fclose(file);

    // printf("Frequency: %d\n", freq);
    return freq;
}

int get_swap() {
    const char* filename = "/proc/meminfo";
    const char* pattern = "MemAvailable:";
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return -1;
    }

    char line[256];
    int mem = -1;

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, pattern) == line) {
            char* value = strtok(line, " \t");
            value = strtok(NULL, " \t");
            mem = atoi(value);
            break;
        }
    }

    fclose(file);

    return mem;
}

int set_governor(std::string target_governor) {
    const char* big_cpu = "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor";
    const char* little_cpu = "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor";

    FILE* file_big = fopen(big_cpu, "w");
    FILE* file_little = fopen(little_cpu, "w");
    if (file_big == NULL || file_little == NULL) {
        printf("Failed to open file: %s or %s \n", big_cpu, little_cpu);
        return -1;
    }

    fprintf(file_big, "%s", target_governor.c_str());
    fprintf(file_little, "%s", target_governor.c_str());

    fclose(file_big);
    fclose(file_little);

    return 0;
}

int set_freq(int big_freq, int little_freq) {
    const char* little_cpu = "/sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed";
    const char* big_cpu = "/sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed";

    FILE* file_big = fopen(big_cpu, "w");
    FILE* file_little = fopen(little_cpu, "w");

    if (file_big == NULL || file_little == NULL) {
        printf("Failed to open file: %s or %s \n", big_cpu, little_cpu);
        return -1;
    }

    fprintf(file_big, "%d", big_freq);
    fprintf(file_little, "%d", little_freq);

    fclose(file_big);
    fclose(file_little);

    printf("freq :  %d, %d\n", big_freq, little_freq);
    return 0;
}

std::vector<int> split_to_int(const std::string& input, char delimiter) {
    std::vector<int> tokens;
    std::string token;
    std::istringstream tokenStream(input);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(stoi(token));
    }

    return tokens;
}

int perf_event_open(struct perf_event_attr* hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int open_perf_event(uint64_t type, uint64_t config) {
    struct perf_event_attr pe;
    std::memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    int fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (fd == -1) {
        std::cerr << "Error opening perf event: " << strerror(errno) << std::endl;
    }
    return fd;
}

int main(int argc, char* argv[]) {
    int server_fd, client_fd, valread;
    struct sockaddr_in server_addr;
    char buffer[1024] = {0};
    if (argc != 2) {
        printf("usage: ./server <view_name>\n");
    }
    std::string view = argv[1];

    // 创建Socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // 配置服务器地址
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // 绑定Socket到端口
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // 监听连接请求
    if (listen(server_fd, 3) < 0) {
        perror("listen failed");
        exit(EXIT_FAILURE);
    }

    printf("服务端已启动，等待客户端连接...\n");


    // State current_state;
    // current_state.init();

    // std::string pid_list = current_state.pid_list;
    // std::vector<int> big_freq_list = split_to_int(current_state.config["BIG_FREQ_LIST"], ',');
    // std::vector<int> little_freq_list = split_to_int(current_state.config["LITTLE_FREQ_LIST"], ',');

    // 初始化
    CPUControl control;
    initialize_cpu_control(&control);
    double utilization[MAX_CPU_COUNT];

    FPSGet* fps = new FPSGet(view.c_str());
    fps->start();

    int big_freq;
    int little_freq;
    int cur_fps;
    int mem;

    double little_util;
    double big_util;
    // 初始化 PMU 事件
    int instructions_fd = open_perf_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    int cycles_fd = open_perf_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    int cache_misses_fd = open_perf_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    if (instructions_fd == -1 || cycles_fd == -1 || cache_misses_fd == -1) {
        return -1;
    }
    std::ofstream data_file("output_data.csv");
    if (!data_file.is_open()) {
        std::cerr << "Failed to open output file!" << std::endl;
        return -1;
    }
    // 接受客户端连接请求
    set_governor("userspace");
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_addr_len = sizeof(client_addr);

        if ((client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_addr_len)) < 0) {
            perror("accept failed");
            exit(EXIT_FAILURE);
        }

        memset(buffer, 0, sizeof(buffer));

        if (read(client_fd, buffer, sizeof(buffer)) < 0) {
            std::cerr << "Failed to read from socket\n";
            return 1;
        }
        printf("客户端已连接：%s\n", inet_ntoa(server_addr.sin_addr));
        printf("客户端的信息为:%s\n", buffer);

        int flag = 0;
        int big_freq, little_freq;
        sscanf(buffer, "%d,%d,%d", &flag, &big_freq, &little_freq);

        if (flag == 0) {
            auto start = std::chrono::high_resolution_clock::now();
            big_freq = get_big_cpu_freq();
            little_freq = get_little_cpu_freq();
            int cur_fps = fps->getFPS();
            int mem = get_swap();

            update_cpu_utilization(&control, utilization);

            little_util = 0.0;
            big_util = 0.0;

            for (int i = 0; i < MAX_CPU_COUNT; i++) {
                std::cout << utilization[i] << "," << i << std::endl;
                if (i < 4) {
                    little_util += utilization[i];
                } else if (i < 7) {
                    big_util += utilization[i];
                }
            }

            ioctl(instructions_fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(cycles_fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(cache_misses_fd, PERF_EVENT_IOC_RESET, 0);

            ioctl(instructions_fd, PERF_EVENT_IOC_ENABLE, 0);
            ioctl(cycles_fd, PERF_EVENT_IOC_ENABLE, 0);
            ioctl(cache_misses_fd, PERF_EVENT_IOC_ENABLE, 0);

            // 执行计算或其他操作

            // 禁用 PMU 计数器
            ioctl(instructions_fd, PERF_EVENT_IOC_DISABLE, 0);
            ioctl(cycles_fd, PERF_EVENT_IOC_DISABLE, 0);
            ioctl(cache_misses_fd, PERF_EVENT_IOC_DISABLE, 0);

            // 读取 PMU 计数器
            uint64_t instructions, cycles, cache_misses;
            read(instructions_fd, &instructions, sizeof(uint64_t));
            read(cycles_fd, &cycles, sizeof(uint64_t));
            read(cache_misses_fd, &cache_misses, sizeof(uint64_t));

            // 计算 IPC 和缓存未命中率
            double ipc = static_cast<double>(instructions) / static_cast<double>(cycles);
            double cache_miss_rate = static_cast<double>(cache_misses) / static_cast<double>(instructions);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;


            std::string data = std::to_string(big_freq) + "," +
                               std::to_string(little_freq) + "," +
                               std::to_string(cur_fps) + "," +
                               std::to_string(mem) + "," +
                               std::to_string(little_util) + "," +
                               std::to_string(big_util) + "," +
                               std::to_string(ipc) + "," +
                               std::to_string(cache_miss_rate);
            send(client_fd, data.c_str(), data.length(), 0);
            // data_file << data << "\n";
            // std::cout << "Data written: " << data << "\n";
            // data_file.flush();
        } else if (flag == 1) {
            int result = set_freq(big_freq, little_freq);
            std::string data = std::to_string(result);
            send(client_fd, data.c_str(), data.length(), 0);
        }
    }
    close(instructions_fd);
    close(cycles_fd);
    close(cache_misses_fd);

    return 0;
}