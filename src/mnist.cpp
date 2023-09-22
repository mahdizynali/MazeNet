#include "include/mnist.hpp"

readUbyte :: readUbyte (const string & image, const string & label) {
    imagePath = image;
    labelPath = label;
}

int readUbyte :: reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

cv::Mat readUbyte :: readImages() {
    ifstream file(imagePath, ios::binary);
    if (!file) {
        cerr << "Failed to open MNIST image file: " << imagePath << endl;
        exit(1);
    }

    int magicNumber, numImages, numRows, numCols;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    file.read((char*)&numImages, sizeof(numImages));
    file.read((char*)&numRows, sizeof(numRows));
    file.read((char*)&numCols, sizeof(numCols));

    magicNumber = reverseInt(magicNumber);
    numImages = reverseInt(numImages);
    numRows = reverseInt(numRows);
    numCols = reverseInt(numCols);

    Mat mnistImages(numImages, numRows * numCols, CV_32FC1);

    for (int i = 0; i < numImages; ++i) {
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                uint8_t pixelValue;
                file.read((char*)&pixelValue, sizeof(pixelValue));
                mnistImages.at<uint8_t>(i, r * numCols + c) = pixelValue;
            }
        }
    }

    return mnistImages;
}

cv::Mat readUbyte :: readLabels() {
    ifstream file(labelPath, ios::binary);
    if (!file) {
        cerr << "Failed to open MNIST label file: " << labelPath << endl;
        exit(1);
    }

    int magicNumber, numLabels;
    file.read((char*)&magicNumber, sizeof(magicNumber));
    file.read((char*)&numLabels, sizeof(numLabels));

    magicNumber = reverseInt(magicNumber);
    numLabels = reverseInt(numLabels);

    cv::Mat mnistLabels(numLabels, 1, CV_32FC1);

    for (int i = 0; i < numLabels; ++i) {
        uint8_t label;
        file.read((char*)&label, sizeof(label));
        mnistLabels.at<uint8_t>(i, 0) = label;
    }

    return mnistLabels;
}