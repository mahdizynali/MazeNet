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

int main(){

    mazeNet maze(784, 128, 10);

    return 0;
}