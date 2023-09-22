#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

class utils{
    public:
        cv::Mat sigmoid (const cv::Mat &);
        cv::Mat softmax (const cv::Mat &);
        cv::Mat relu (const cv::Mat &);
        cv::Mat dot (const cv::Mat &, const cv::Mat &);
        cv::Mat sum (const cv::Mat &, const cv::Mat &);
};

cv::Mat utils :: sigmoid (const cv::Mat & X) {
    cv::Mat sigmoidResult;
    cv::exp(-X, sigmoidResult);
    cv::add(cv::Scalar(1.0), sigmoidResult, sigmoidResult);
    cv::divide(cv::Scalar(1.0), sigmoidResult, sigmoidResult);

    return sigmoidResult;
}

cv::Mat utils :: softmax (const cv::Mat & X) {
    cv::Mat expX;
    cv::exp(X, expX);

    cv::Mat sumExpX;
    cv::reduce(expX, sumExpX, 1, cv::REDUCE_SUM, CV_32FC1);

    cv::Mat softmaxResult;
    cv::divide(expX, repeat(sumExpX, 1, X.cols), softmaxResult);

    return softmaxResult;
}

cv::Mat utils :: relu (const cv::Mat & X) {
    // cv::Mat tmp = X;
    // for (int i=0; i<tmp.rows; i++){
    //     for(int j=0; j<tmp.cols; j++){
    //         if(tmp.at<uchar>(i, j) > 0)
    //             continue;
    //         else
    //             tmp.at<uchar>(i, j) = 0;
    //     }
    // }
    // return tmp;

    // simplified
    cv::Mat tmp;
    cv::max(X, cv::Scalar(0.0f), tmp);
    return tmp;
}

cv::Mat utils :: sum (const cv::Mat & mat1, const cv::Mat & mat2) {
    // assert(mat1.rows == mat2.rows && mat1.cols == mat2.cols);

    // cv::Mat tmp(mat1.rows, mat1.cols, CV_32FC1);

    // for (int i = 0; i < mat1.rows; i++) {
    //     for (int j = 0; j < mat1.cols; j++) {
    //         tmp.at<float>(i, j) = mat1.at<float>(i, j) + mat2.at<float>(i, j);
    //     }
    // }
    // return tmp;

    //simplified
    cv::Mat tmp;
    cv::add(mat1, mat2, tmp); 
    return tmp;
}

cv::Mat utils :: dot (const cv::Mat & mat1, const cv::Mat & mat2) {
    assert(mat1.cols == mat2.rows);

    cv::Mat tmp(mat1.rows, mat2.cols, mat1.type());

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat2.cols; j++) {
            tmp.at<float>(i, j) = 0;

            for (int z = 0; z < mat1.cols; z++) {
                tmp.at<float>(i, j) += mat1.at<float>(i, z) * mat2.at<float>(z, j);
            }
        }
    }
    return tmp;
}

class mazeNet{

    private :
        int input_size, output_size, hidden_size;
        cv::Mat w1, b1, w2, b2;
        cv::Mat z1, a1, z2, result;

    public :
        mazeNet(int, int, int);
        cv::Mat forward(const cv::Mat &);
        void backward(const cv::Mat &, const cv::Mat &, const cv::Mat &, float);
        void printLayerSize();
};

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

cv::Mat mazeNet :: forward (const cv::Mat & X){
    z1 = utils().dot(X, w1) + b1;
    // z1 = utils().sum(z1, b1);

    a1 = utils().relu(z1);

    z2 = utils().dot(a1, w2) + b2;
    // z2 = utils().sum(z2, b2);

    result = utils().softmax(z2);

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
    cv::Mat reluGradient = utils().relu(z1); // Gradient of the relu function
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

int main(){

    mazeNet maze(784, 128, 10);

    maze.printLayerSize();

    return 0;
}