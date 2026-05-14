#include "include/model.hpp"

#include <cmath>
#include <ctime>

cv::Mat randomNormal(int rows, int cols) {
    /*
        He initialization for ReLU networks.

        rows = fan_in
        cols = fan_out
    */

    cv::Mat result(rows, cols, CV_64FC1);

    static cv::RNG rng(static_cast<uint64>(cv::getTickCount()));

    double scale = std::sqrt(2.0 / static_cast<double>(rows));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at<double>(i, j) = rng.gaussian(scale);
        }
    }

    return result;
}

mazeNet::mazeNet(int in, int hide, int out) {
    this->input_size = in;
    this->hidden_size = hide;
    this->output_size = out;

    w1 = randomNormal(input_size, hidden_size);
    b1 = cv::Mat::zeros(1, hidden_size, CV_64FC1);

    w2 = randomNormal(hidden_size, output_size);
    b2 = cv::Mat::zeros(1, output_size, CV_64FC1);
}

cv::Mat mazeNet::forward(const cv::Mat& X) {
    CV_Assert(X.type() == CV_64FC1);
    CV_Assert(X.cols == input_size);

    /*
        Layer 1:
            z1 = X . w1 + b1
            a1 = ReLU(z1)

        Layer 2:
            z2 = a1 . w2 + b2
            result = Softmax(z2)
    */

    z1 = utils.dot(X, w1);
    z1 = utils.sum(z1, b1);

    a1 = utils.relu(z1);

    z2 = utils.dot(a1, w2);
    z2 = utils.sum(z2, b2);

    result = utils.softmax(z2);

    return result;
}

void mazeNet::backward(
    const cv::Mat& X_train,
    const cv::Mat& y_train,
    const cv::Mat& y_pred,
    float learning_rate
) {
    CV_Assert(X_train.type() == CV_64FC1);
    CV_Assert(y_train.type() == CV_64FC1);
    CV_Assert(y_pred.type() == CV_64FC1);

    CV_Assert(X_train.rows == y_train.rows);
    CV_Assert(X_train.rows == y_pred.rows);
    CV_Assert(X_train.cols == input_size);
    CV_Assert(y_train.cols == 1);
    CV_Assert(y_pred.cols == output_size);

    const double m = static_cast<double>(X_train.rows);
    const double lr = static_cast<double>(learning_rate);

    /*
        Convert labels:

        y_train:
            batch_size x 1

        y_true:
            batch_size x output_size
    */

    cv::Mat y_true = utils.oneHot(y_train, output_size);

    /*
        Softmax + categorical cross entropy gradient:

        dZ2 = y_pred - y_true
    */

    cv::Mat lossGradient = utils.sub(y_pred, y_true);

    /*
        Output layer gradients:

        dW2 = a1.T . dZ2 / m
        db2 = mean(dZ2)
    */

    cv::Mat a1Transpose = a1.t();
    cv::Mat w2Gradient = utils.dot(a1Transpose, lossGradient) / m;

    cv::Mat b2Gradient;
    cv::reduce(lossGradient, b2Gradient, 0, cv::REDUCE_AVG, CV_64FC1);

    /*
        Hidden layer gradients:

        dA1 = dZ2 . w2.T
        dZ1 = dA1 * ReLU'(z1)
        dW1 = X.T . dZ1 / m
        db1 = mean(dZ1)
    */

    cv::Mat z1Gradient = utils.dot(lossGradient, w2.t());

    cv::Mat reluGradient = utils.reluDerivative(z1);
    z1Gradient = z1Gradient.mul(reluGradient);

    cv::Mat XTranspose = X_train.t();
    cv::Mat w1Gradient = utils.dot(XTranspose, z1Gradient) / m;

    cv::Mat b1Gradient;
    cv::reduce(z1Gradient, b1Gradient, 0, cv::REDUCE_AVG, CV_64FC1);

    /*
        Parameter update
    */

    w1 -= lr * w1Gradient;
    b1 -= lr * b1Gradient;

    w2 -= lr * w2Gradient;
    b2 -= lr * b2Gradient;
}

void mazeNet::printLayerSize() {
    cout << "input size  : " << input_size << endl;
    cout << "hidden size : " << hidden_size << endl;
    cout << "output size : " << output_size << endl;
}
