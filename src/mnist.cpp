#include "include/mnist.hpp"

readUbyte :: readUbyte (const string & image, const string & label) {
    imagePath = image;
    labelPath = label;
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

    Mat mnistImages(numImages, numRows * numCols, CV_8U);

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

    Mat mnistLabels(numLabels, 1, CV_8U);

    for (int i = 0; i < numLabels; ++i) {
        uint8_t label;
        file.read((char*)&label, sizeof(label));
        mnistLabels.at<uint8_t>(i, 0) = label;
    }

    return mnistLabels;
}