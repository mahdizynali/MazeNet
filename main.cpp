#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

class utils{
    public:
        cv::Mat sigmoid (const cv::Mat &);
        cv::Mat softmax (const cv::Mat &);
        cv::Mat relu (const cv::Mat &);
        cv::Mat dot (const cv::Mat &, const cv::Mat &);
};

cv::Mat utils :: dot (const cv::Mat &mat1, const cv::Mat &mat2) {
    assert(mat1.cols == mat2.rows);

    cv::Mat tmp(mat1.rows, mat2.cols, mat1.type());

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat2.cols; j++) {
            tmp.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);

            for (int z = 0; z < mat1.cols; z++) {
                for (int c = 0; c < 3; c++) {
                    tmp.at<cv::Vec3b>(i, j)[c] += mat1.at<cv::Vec3b>(i, z)[c] * mat2.at<cv::Vec3b>(z, j)[c];
                }
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
    randu(w1, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255)); // initiate random weigths

    b1 = cv::Mat::zeros(1, hidden_size, CV_32FC1);

    w2 = cv::Mat (hidden_size, output_size, CV_32FC1);
    randu(w2, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    b2 = cv::Mat::zeros(2, output_size, CV_32FC1);

}

cv::Mat mazeNet :: forward (const cv::Mat & X){
    z1 = utils().dot(w1, w1) + b1;
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