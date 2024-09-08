#include <string>
#include <deque>
#include <mutex>
#include <vector>
#include <thread>

class FPSGet {
private:
    std::string view;
    int fps;
    long t;
    std::deque<unsigned long long> frame_queue;
    std::mutex lock;
    unsigned long long base_timestamp = 0;
    unsigned long long last_timestamp = 0;
    void getFrameDataThread();
    std::thread m_fps_thread;

public:
    bool while_flag = 1;
    FPSGet(const std::string& view);
    void start();
    void stop();
    int getFPS();
    std::pair<unsigned long long, std::vector<unsigned long long>> getFrameData();
};

std::string execute(const std::string& command);

const int MAX_QUEUE_SIZE = 200; // 设置队列的最大长度

#define MAX_CPU_COUNT 8
#define MAX_LINE_LENGTH 256
#define TARGET_FPS 60