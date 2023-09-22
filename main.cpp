#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

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

    w1 = cv::Mat (input_size, hidden_size, CV_8UC3);
    randu(w1, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255)); // initiate random weigths

    b1 = Mat::zeros(1, hidden_size, CV_8UC1);

    w2 = cv::Mat (hidden_size, output_size, CV_8UC3);
    randu(w2, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    b2 = Mat::zeros(2, output_size, CV_8UC1);

}

cv::Mat mazeNet :: forward (const cv::Mat & X){

}

void mazeNet :: printLayerSize () {
    cout << "input size : " << input_size << endl;
    cout << "hiden size : " << hidden_size << endl;
    cout << "output size : " << output_size << endl;
}

int main(){

    mazeNet maze(784, 128, 10);

    return 0;
}