#ifndef MNIST_HPP
#define MNIST_HPP

#include "config.hpp"

class readUbyte{

    private:
        const string & labelPath;
        const string & imagePath;

    public:
        readUbyte(const string &, const string &);
        cv::Mat readImages(const string &);
        cv::Mat readLabels(const string &);
};

#endif // MNIST_HPP