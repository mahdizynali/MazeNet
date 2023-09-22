#ifndef MNIST_HPP
#define MNIST_HPP

#include "config.hpp"

class readUbyte{

    private:
        string labelPath;
        string imagePath;

    public:
        readUbyte(const string &, const string &);
        cv::Mat readImages();
        cv::Mat readLabels();
        int reverseInt(int);
};

#endif // MNIST_HPP