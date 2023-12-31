#include "include/utils.hpp"

cv::Mat helper::sigmoid (const cv::Mat & X) {
    cv::Mat sigmoidResult;
    cv::exp(-X, sigmoidResult);
    cv::add(cv::Scalar(1.0), sigmoidResult, sigmoidResult);
    cv::divide(cv::Scalar(1.0), sigmoidResult, sigmoidResult);

    return sigmoidResult;
}

cv::Mat helper :: softmax(const cv::Mat &X) {

    cv::Mat maxVal;
    cv::reduce(X, maxVal, 1, cv::REDUCE_MAX, CV_64FC1);

    cv::Mat expX;
    cv::exp(X - repeat(maxVal, 1, X.cols), expX);

    cv::Mat sumExpX;
    cv::reduce(expX, sumExpX, 1, cv::REDUCE_SUM, CV_64FC1);

    cv::Mat softmaxResult = expX / repeat(sumExpX, 1, X.cols);

    return softmaxResult;
}

// cv::Mat helper::softmax(const cv::Mat& X) {
//     cv::Mat tmp(X.rows, X.cols, CV_64FC1);
//     double exp_sum = 0;
//     std::vector<double> expResults;

//     for (int i = 0; i < X.rows; ++i) {
//         for (int j = 0; j < X.cols; ++j) {
//             double exp_value = expf(X.at<float>(i, j));
//             cout<<exp_value<<endl;
//             expResults.push_back(exp_value);
//             exp_sum += exp_value;
//         }
//     }

//     int index = 0;
//     for (int i = 0; i < X.rows; ++i) {
//         for (int j = 0; j < X.cols; ++j) {
//             tmp.at<float>(i, j) = static_cast<float>(expResults[index] / exp_sum);
//             ++index;
//         }
//     }
//     return tmp;
// }


cv::Mat helper::relu (const cv::Mat & X) {
    cv::Mat tmp(X.rows, X.cols, CV_64FC1);
    for (int i=0; i<tmp.rows; i++){
        for(int j=0; j<tmp.cols; j++){
            if(tmp.at<float>(i, j) > 0.0)
                continue;
            else
                tmp.at<float>(i, j) = 0.0;
        }
    }
    return tmp;

    // simplified
    // cv::Mat tmp;
    // cv::max(X, cv::Scalar(0.0f), tmp);
    // return tmp;
}

cv::Mat helper::sum (const cv::Mat & mat1, const cv::Mat & mat2) {

    cv::Mat tmp(mat1.rows, mat1.cols, CV_64FC1);

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.cols; j++) {
            tmp.at<float>(i, j) = mat1.at<float>(i, j) + mat2.at<float>(0, j);
        }
    }

    return tmp;

    //simplified
    // cv::Mat tmp;
    // cv::add(mat1, mat2, tmp); 
    // return tmp;
}

cv::Mat helper::sub (const cv::Mat & mat1, const cv::Mat & mat2) {
    assert(mat1.rows == mat2.rows);

    cv::Mat tmp(mat1.rows, mat1.cols, CV_64FC1);

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

cv::Mat helper::dot (const cv::Mat & mat1, const cv::Mat & mat2) {

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


double helper::categoricalCrossEntropy (const cv::Mat& y_train, const cv::Mat& y_pred) {

    int num_samples = y_train.rows;
    int num_classes = 10;

    double loss = 0.0;
    double epsilon = 1e-15;

    cv::Mat y_train_one_hot_labels(num_samples, num_classes, CV_64FC1, cv::Scalar(0.0));

    for (int i = 0; i < num_samples; ++i) {
        int true_label = static_cast<int>(y_train.at<float>(i, 0));
        y_train_one_hot_labels.at<float>(i, true_label) = 1.0;
    }

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