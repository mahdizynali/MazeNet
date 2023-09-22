#ifndef UTILS_HPP
#define UTILS_HPP

#include "config.hpp"

class helper{

    public:
        cv::Mat sigmoid (const cv::Mat &);
        cv::Mat softmax (const cv::Mat &);
        cv::Mat relu (const cv::Mat &);
        cv::Mat dot (const cv::Mat &, const cv::Mat &);
        cv::Mat sum (const cv::Mat &, const cv::Mat &);
        cv::Mat sub (const cv::Mat &, const cv::Mat &);
        double categoricalCrossEntropy (const cv::Mat &, const cv::Mat &);
};

#endif // UTILS_HPP