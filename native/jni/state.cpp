#include "state.h"

std::string get_package_name(const std::string& view) {
    return "com.bilibili.app.in";
    std::size_t slashPos = view.find('/');
    if (slashPos != std::string::npos) {
        return view.substr(0, slashPos);
    }
    return "";
}


std::string get_view() {
    return "SurfaceView\\[com\\.bilibili\\.app\\.in/com\\.bilibili\\.video\\.videodetail\\.VideoDetailsActivity\\]\\(BLAST\\)\\#169";
    std::string out = execute("dumpsys SurfaceFlinger | grep -i focus -A 10");
    std::vector<std::string> a;
    std::string view = "";

    // 将输出拆分为行
    size_t start = 0;
    size_t end = out.find("\n");
    while (end != std::string::npos) {
        a.push_back(out.substr(start, end - start));
        start = end + 1;
        end = out.find("\n", start);
    }

    // 查找带有 '*' 的行，并获取其前一行作为 view
    for (size_t index = 0; index < a.size(); index++) {
        std::cout << a[index] << std::endl;
        if (a[index][a[index].length() - 2] == '*') {
            view = a[index - 1];
            break;
        }
    }

    view = view.substr(view.find_first_not_of(" \t") , view.find_last_not_of(" \t") + 1);  // 去除尾部空格

    out = execute("dumpsys SurfaceFlinger --list");
    a.clear();

    // 将输出拆分为行
    start = 0;
    end = out.find("\n");
    while (end != std::string::npos) {
        a.push_back(out.substr(start, end - start));
        start = end + 1;
        end = out.find("\n", start);
    }

    // 构建正则表达式模式
    std::string escaped_text = std::regex_replace(view, std::regex(R"([.*+?^${}()|\[\]\/\\])"), R"(\$&)");
    std::string pattern = std::regex_replace(escaped_text, std::regex(R"(\[...\])"), ".*?");
    std::cout << pattern << std::endl;

    // 在输出中查找匹配结果
    std::string result;
    std::regex re(pattern);
    for (size_t i = 0; i < a.size(); i++) {
        std::smatch match;
        if (std::regex_search(a[i], match, re)) {
            result = match.str();
            break;
        }
    }

    return result;
}

std::string get_pid_list(std::string package_name) {
    std::string str = execute("pidof " + package_name);
    std::string pid1 = str.substr(0, str.length() - 1);
    str = execute("ps -T -p " + pid1 + " | grep RenderThread");
    std::istringstream iss(str);
    std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                    std::istream_iterator<std::string>{}};

    std::string pid2 = tokens[2];
    str = execute("pidof surfaceflinger");
    std::string pid3 = str.substr(0, str.length() - 1);
    return pid1 + "," + pid2 + "," + pid3;
}

int State::init() {
    std::cout << "View: " << this->view << std::endl;
    this->package_name = get_package_name(view);
    std::cout << "Package : " << this->package_name << std::endl;
    this->pid_list = get_pid_list(package_name);

    std::ifstream file("/data/local/tmp/config.ini");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            // 忽略空行和注释
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // 解析键值对
            size_t delimiterPos = line.find('=');
            if (delimiterPos != std::string::npos) {
                std::string key = line.substr(0, delimiterPos);
                std::string value = line.substr(delimiterPos + 1);
                config[key] = value;
            }
        }
        file.close();
    } else {
        std::cout << "Failed to open the configuration file." << std::endl;
        return -1;
    }
    return 0;
}