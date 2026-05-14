#include "include/utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

cv::Mat helper::sigmoid(const cv::Mat& X) {
    CV_Assert(X.type() == CV_64FC1);

    cv::Mat sigmoidResult;
    cv::exp(-X, sigmoidResult);
    cv::add(cv::Scalar(1.0), sigmoidResult, sigmoidResult);
    cv::divide(cv::Scalar(1.0), sigmoidResult, sigmoidResult);

    return sigmoidResult;
}

cv::Mat helper::softmax(const cv::Mat& X) {
    CV_Assert(X.type() == CV_64FC1);

    /*
        Stable row-wise softmax.

        Input shape:
            batch_size x num_classes

        Output shape:
            batch_size x num_classes
    */

    cv::Mat maxVal;
    cv::reduce(X, maxVal, 1, cv::REDUCE_MAX, CV_64FC1);

    cv::Mat shifted = X - cv::repeat(maxVal, 1, X.cols);

    cv::Mat expX;
    cv::exp(shifted, expX);

    cv::Mat sumExpX;
    cv::reduce(expX, sumExpX, 1, cv::REDUCE_SUM, CV_64FC1);

    cv::Mat softmaxResult = expX / cv::repeat(sumExpX, 1, X.cols);

    return softmaxResult;
}

cv::Mat helper::relu(const cv::Mat& X) {
    CV_Assert(X.type() == CV_64FC1);

    cv::Mat out(X.rows, X.cols, CV_64FC1);

    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            double value = X.at<double>(i, j);
            out.at<double>(i, j) = std::max(0.0, value);
        }
    }

    return out;
}

cv::Mat helper::reluDerivative(const cv::Mat& X) {
    CV_Assert(X.type() == CV_64FC1);

    cv::Mat out(X.rows, X.cols, CV_64FC1);

    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
            out.at<double>(i, j) = X.at<double>(i, j) > 0.0 ? 1.0 : 0.0;
        }
    }

    return out;
}

cv::Mat helper::oneHot(const cv::Mat& labels, int numClasses) {
    CV_Assert(labels.type() == CV_64FC1);
    CV_Assert(labels.cols == 1);
    CV_Assert(numClasses > 0);

    cv::Mat oneHotLabels = cv::Mat::zeros(labels.rows, numClasses, CV_64FC1);

    for (int i = 0; i < labels.rows; i++) {
        int label = static_cast<int>(labels.at<double>(i, 0));

        if (label < 0 || label >= numClasses) {
            throw std::out_of_range("Label value is outside valid class range.");
        }

        oneHotLabels.at<double>(i, label) = 1.0;
    }

    return oneHotLabels;
}

cv::Mat helper::sum(const cv::Mat& mat1, const cv::Mat& mat2) {
    /*
        Adds bias row vector to every row of mat1.

        mat1 shape:
            batch_size x features

        mat2 shape:
            1 x features
    */

    CV_Assert(mat1.type() == CV_64FC1);
    CV_Assert(mat2.type() == CV_64FC1);
    CV_Assert(mat2.rows == 1);
    CV_Assert(mat1.cols == mat2.cols);

    cv::Mat out(mat1.rows, mat1.cols, CV_64FC1);

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.cols; j++) {
            out.at<double>(i, j) = mat1.at<double>(i, j) + mat2.at<double>(0, j);
        }
    }

    return out;
}

cv::Mat helper::sub(const cv::Mat& mat1, const cv::Mat& mat2) {
    CV_Assert(mat1.type() == CV_64FC1);
    CV_Assert(mat2.type() == CV_64FC1);
    CV_Assert(mat1.rows == mat2.rows);
    CV_Assert(mat1.cols == mat2.cols);

    cv::Mat out(mat1.rows, mat1.cols, CV_64FC1);

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.cols; j++) {
            out.at<double>(i, j) = mat1.at<double>(i, j) - mat2.at<double>(i, j);
        }
    }

    return out;
}

cv::Mat helper::dot(const cv::Mat& mat1, const cv::Mat& mat2) {
    CV_Assert(mat1.type() == CV_64FC1);
    CV_Assert(mat2.type() == CV_64FC1);
    CV_Assert(mat1.cols == mat2.rows);

    cv::Mat out = cv::Mat::zeros(mat1.rows, mat2.cols, CV_64FC1);

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat2.cols; j++) {
            double value = 0.0;

            for (int k = 0; k < mat1.cols; k++) {
                value += mat1.at<double>(i, k) * mat2.at<double>(k, j);
            }

            out.at<double>(i, j) = value;
        }
    }

    return out;
}

double helper::categoricalCrossEntropy(const cv::Mat& y_train, const cv::Mat& y_pred) {
    /*
        y_train shape:
            batch_size x 1

        y_pred shape:
            batch_size x num_classes

        y_pred must already be softmax output.
    */

    CV_Assert(y_train.type() == CV_64FC1);
    CV_Assert(y_pred.type() == CV_64FC1);
    CV_Assert(y_train.rows == y_pred.rows);
    CV_Assert(y_train.cols == 1);

    const int numSamples = y_train.rows;
    const int numClasses = y_pred.cols;
    const double epsilon = 1e-15;

    cv::Mat yTrue = oneHot(y_train, numClasses);

    double loss = 0.0;

    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numClasses; j++) {
            double trueValue = yTrue.at<double>(i, j);
            double predictedValue = y_pred.at<double>(i, j);

            predictedValue = std::max(epsilon, std::min(1.0 - epsilon, predictedValue));

            loss += -trueValue * std::log(predictedValue);
        }
    }

    return loss / static_cast<double>(numSamples);
}
