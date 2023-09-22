#include "include/utils.hpp"

cv::Mat helper :: sigmoid (const cv::Mat & X) {
    cv::Mat sigmoidResult;
    cv::exp(-X, sigmoidResult);
    cv::add(cv::Scalar(1.0), sigmoidResult, sigmoidResult);
    cv::divide(cv::Scalar(1.0), sigmoidResult, sigmoidResult);

    return sigmoidResult;
}

cv::Mat helper :: softmax (const cv::Mat & X) {
    cv::Mat expX;
    cv::exp(X, expX);

    cv::Mat sumExpX;
    cv::reduce(expX, sumExpX, 1, cv::REDUCE_SUM, CV_32FC1);

    cv::Mat softmaxResult;
    cv::divide(expX, repeat(sumExpX, 1, X.cols), softmaxResult);

    return softmaxResult;
}

cv::Mat helper :: relu (const cv::Mat & X) {
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

cv::Mat helper :: sum (const cv::Mat & mat1, const cv::Mat & mat2) {
    assert(mat1.cols == mat2.cols);

    cv::Mat tmp(mat1.rows, mat1.cols, CV_32FC1);

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.cols; j++) {
            tmp.at<float>(i, j) = mat1.at<float>(i, j) + mat2.at<float>(i, j);
        }
    }

    return tmp;

    //simplified
    // cv::Mat tmp;
    // cv::add(mat1, mat2, tmp); 
    // return tmp;
}

cv::Mat helper :: sub (const cv::Mat & mat1, const cv::Mat & mat2) {
    assert(mat1.rows == mat2.rows);

    cv::Mat tmp(mat1.rows, mat1.cols, CV_32FC1);

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.cols; j++) {
            tmp.at<float>(i, j) = mat1.at<float>(i, j) - mat2.at<float>(i, j);
        }
    }

    return tmp;

    //simplified
    // cv::Mat tmp;
    // cv::subtract(mat1, mat2, tmp); 
    // return tmp;
}

cv::Mat helper :: dot (const cv::Mat & mat1, const cv::Mat & mat2) {

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



double helper::categoricalCrossEntropy(const cv::Mat& y_train, const cv::Mat& y_pred) {
    int num_samples = y_train.rows;
    int num_classes = 10;

    double loss = 0.0;
    double epsilon = 1e-15;

    // Create a matrix for one-hot encoding y_train
    cv::Mat y_train_one_hot_labels(num_samples, num_classes, CV_32FC1, cv::Scalar(0.0));

    for (int i = 0; i < num_samples; ++i) {
        int true_label = static_cast<int>(y_train.at<float>(i, 0));
        y_train_one_hot_labels.at<float>(i, true_label) = 1.0;
    }
    // cout<<"pred : "<<y_pred.rows<<"  "<<y_pred.cols<<endl;
    // cout<<"train : "<<y_train_one_hot_labels.rows<<"  "<<y_train_one_hot_labels.cols<<endl;

    // Compute the cross-entropy loss
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            double y_true = y_train_one_hot_labels.at<float>(i, j);
            double y_predicted = y_pred.at<float>(i, j);
            y_predicted = std::max(epsilon, std::min(1.0 - epsilon, y_predicted));
            loss += -y_true * log(y_predicted);
        }
    }
    return loss / static_cast<double>(batch_size);
}