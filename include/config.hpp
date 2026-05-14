#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

constexpr int in_size = 784;
constexpr int hide_size = 128;
constexpr int out_size = 10;
constexpr double l_rate = 0.005;
constexpr int total_epochs = 10;
constexpr int batch_size = 64;

using namespace std;
using namespace cv;

#endif // CONFIG_HPP
