#include "include/model.hpp"

mazeNet :: mazeNet (int in, int hide, int out) {
    input_size = in;
    hidden_size = hide;
    output_size = out;

    w1 = cv::Mat (input_size, hidden_size, CV_32FC1); // Use CV_32FC1 for floating-point weights
    randu(w1, cv::Scalar(0.0), cv::Scalar(1.0)); // initiate random weigths

    b1 = cv::Mat::zeros(1, hidden_size, CV_32FC1);

    w2 = cv::Mat (hidden_size, output_size, CV_32FC1);
    randu(w2, cv::Scalar(0.0), cv::Scalar(1.0));

    b2 = cv::Mat::zeros(2, output_size, CV_32FC1);

}

cv::Mat mazeNet :: forward (const cv::Mat & X) {
    z1 = utils.dot(X, w1) + b1;
    // z1 = utils.sum(z1, b1);

    a1 = utils.relu(z1);

    z2 = utils.dot(a1, w2) + b2;
    // z2 = utils.sum(z2, b2);

    result = utils.softmax(z2);

    return result;
}

void mazeNet :: backward (const cv::Mat & X_train ,const cv::Mat & y_train, const cv::Mat & y_pred, float learning_rate) {

    cv::Mat lossGradient = y_pred - y_train;

    // Calculate gradients for the output layer
    cv::Mat a1_Transpose = a1.t();
    cv::Mat w2_Gradient = a1_Transpose * lossGradient;
    cv::Mat b2_Gradient = cv::Mat::ones(1, lossGradient.rows, CV_32FC1) * lossGradient;

    // Calculate gradients for the hidden layer
    cv::Mat z1_Gradient = lossGradient * w2.t();
    cv::Mat reluGradient = utils.relu(z1); // Gradient of the relu function
    z1_Gradient = z1_Gradient.mul(reluGradient);

    cv::Mat X_Transpose = X_train.t();
    cv::Mat w1_Gradient = X_Transpose * z1_Gradient;
    cv::Mat b1_Gradient = cv::Mat::ones(1, z1_Gradient.rows, CV_32FC1) * z1_Gradient;

    // Update weights and biases using gradients and learning rate
    w1 -= learning_rate * w1_Gradient;
    b1 -= learning_rate * b1_Gradient;
    w2 -= learning_rate * w2_Gradient;
    b2 -= learning_rate * b2_Gradient;
}

void mazeNet :: printLayerSize () {
    cout << "input size : " << input_size << endl;
    cout << "hiden size : " << hidden_size << endl;
    cout << "output size : " << output_size << endl;
}
