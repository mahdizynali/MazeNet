#include "include/model.hpp"

cv::Mat randomNormal(int rows, int cols) {
    cv::Mat result(rows, cols, CV_32FC1);
    cv::RNG rng(static_cast<unsigned int>(std::time(0)));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value = static_cast<float>(rng.gaussian(1.0));
            result.at<float>(i, j) = value;
        }
    }

    return result;
}

mazeNet::mazeNet(int in, int hide, int out) {
    this->input_size = in;
    this->hidden_size = hide;
    this->output_size = out;

    w1 = randomNormal(input_size, hidden_size);
    b1 = cv::Mat::zeros(1, hidden_size, CV_32FC1);

    w2 = randomNormal(hidden_size, output_size);
    b2 = cv::Mat::zeros(1, output_size, CV_32FC1);
}


cv::Mat mazeNet :: forward (const cv::Mat & X) {
    z1 = utils.dot(X, w1);
    // z1 = X * w1;
    z1 = utils.sum(z1, b1);

    a1 = utils.relu(z1);
    cout<<"before :: "<<a1<<endl;

    z2 = utils.dot(a1, w2);
    z2 = utils.sum(z2, b2);
    result = utils.softmax(z2);
    return result;
}

void mazeNet :: backward (const cv::Mat & X_train ,const cv::Mat & y_train, const cv::Mat & y_pred, float learning_rate) {

    cv::Mat y_pred_one_hot(y_pred.rows, y_pred.cols, CV_32FC1, cv::Scalar(0.0));  // Initialize with zeros

    for (int i = 0; i < y_pred.rows; i++) {
        // Find the index of the maximum value in the current row (argmax)
        cv::Point max_index;
        cv::minMaxLoc(y_pred.row(i), nullptr, nullptr, nullptr, &max_index);
        // Set the corresponding class to 1
        y_pred_one_hot.at<float>(i, max_index.x) = 1.0;
    }

    cv::Mat lossGradient = utils.sub(y_pred, y_train);

    // Calculate gradients for the output layer
    cv::Mat a1_Transpose = a1.t();
    cv::Mat w2_Gradient = utils.dot(a1_Transpose ,lossGradient);
    cv::Mat b2_Gradient = utils.dot(cv::Mat::ones(1, lossGradient.rows, CV_32FC1) ,lossGradient);

    // Calculate gradients for the hidden layer
    cv::Mat z1_Gradient = utils.dot(lossGradient ,w2.t());
    cv::Mat reluGradient = utils.relu(z1); // Gradient of the relu function
    z1_Gradient = z1_Gradient.mul(reluGradient);

    cv::Mat X_Transpose = X_train.t();
    cv::Mat w1_Gradient = utils.dot(X_Transpose ,z1_Gradient);
    cv::Mat b1_Gradient = utils.dot(cv::Mat::ones(1, z1_Gradient.rows, CV_32FC1) ,z1_Gradient);
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
