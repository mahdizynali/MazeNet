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
        void backward();
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
    z1 = utils().dot(X, w1);
    z1 = utils().sum(z1, b1);

    a1 = utils().relu(z1);

    z2 = utils().dot(a1, w2) + b2;
    return result;
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