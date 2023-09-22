#ifndef MODEL_HPP
#define MODEL_HPP

#include "config.hpp"
#include "utils.hpp"

class mazeNet{

    private :
        int input_size, output_size, hidden_size;
        cv::Mat z1, a1, z2, result;
        helper utils;

    public :
        cv::Mat w1, b1, w2, b2;
        mazeNet(int, int, int);
        cv::Mat forward(const cv::Mat &);
        void backward(const cv::Mat &, const cv::Mat &, const cv::Mat &, float);
        void printLayerSize();
};

#endif // MODEL_HPP