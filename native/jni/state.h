#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "execute.h"

class State {
   public:
    int init();
    std::string view;
    std::string package_name;
    std::string pid_list;
    std::unordered_map<std::string, std::string> config;
};