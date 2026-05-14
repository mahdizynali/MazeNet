#include "include/config.hpp"
#include "include/utils.hpp"
#include "include/model.hpp"
#include "include/mnist.hpp"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

cv::Mat X_train;
cv::Mat y_train;
cv::Mat X_test;
cv::Mat y_test;

bool fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

int argmaxRow(const cv::Mat& row) {
    CV_Assert(row.rows == 1);

    cv::Point maxLoc;
    cv::minMaxLoc(row, nullptr, nullptr, nullptr, &maxLoc);

    return maxLoc.x;
}

void shuffleDataset(cv::Mat& X, cv::Mat& y) {
    CV_Assert(X.rows == y.rows);

    std::vector<int> indices(X.rows);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    cv::Mat X_shuffled(X.rows, X.cols, X.type());
    cv::Mat y_shuffled(y.rows, y.cols, y.type());

    for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        X.row(indices[i]).copyTo(X_shuffled.row(i));
        y.row(indices[i]).copyTo(y_shuffled.row(i));
    }

    X = X_shuffled;
    y = y_shuffled;
}

void loadDataset() {
    const std::string trainImagesPath = "../dataset/train-images.idx3-ubyte";
    const std::string trainLabelsPath = "../dataset/train-labels.idx1-ubyte";

    const std::string testImagesPath = "../dataset/t10k-images.idx3-ubyte";
    const std::string testLabelsPath = "../dataset/t10k-labels.idx1-ubyte";

    readUbyte trainDataset(trainImagesPath, trainLabelsPath);

    cv::Mat X_train_full = trainDataset.readImages();
    cv::Mat y_train_full = trainDataset.readLabels();

    X_train_full.convertTo(X_train_full, CV_64FC1);
    y_train_full.convertTo(y_train_full, CV_64FC1);

    /*
        Best case:
        Use the official MNIST test files.

        Fallback:
        If t10k files are missing, split train data into 80% train and 20% test.
    */

    if (fileExists(testImagesPath) && fileExists(testLabelsPath)) {
        readUbyte testDataset(testImagesPath, testLabelsPath);

        X_train = X_train_full.clone();
        y_train = y_train_full.clone();

        X_test = testDataset.readImages();
        y_test = testDataset.readLabels();

        X_test.convertTo(X_test, CV_64FC1);
        y_test.convertTo(y_test, CV_64FC1);

        std::cout << "Loaded official MNIST train/test files." << std::endl;
    } else {
        double trainPercent = 0.8;
        int numSamples = X_train_full.rows;
        int splitIndex = static_cast<int>(trainPercent * numSamples);

        X_train = X_train_full.rowRange(0, splitIndex).clone();
        y_train = y_train_full.rowRange(0, splitIndex).clone();

        X_test = X_train_full.rowRange(splitIndex, numSamples).clone();
        y_test = y_train_full.rowRange(splitIndex, numSamples).clone();

        std::cout << "Official test files not found. Used 80/20 train split." << std::endl;
    }

    shuffleDataset(X_train, y_train);

    std::cout << "Train samples: " << X_train.rows << std::endl;
    std::cout << "Test samples : " << X_test.rows << std::endl;
}

void displayRandom() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < 5; i++) {
        int randomIndex = std::rand() % X_train.rows;

        cv::Mat image = X_train.row(randomIndex).reshape(1, 28).clone();

        /*
            MNIST data is usually normalized between 0 and 1.
            Convert to 8-bit for better OpenCV display.
        */

        cv::Mat displayImage;
        image.convertTo(displayImage, CV_8UC1, 255.0);

        int label = static_cast<int>(y_train.at<double>(randomIndex, 0));

        cv::putText(
            displayImage,
            std::to_string(label),
            cv::Point(10, 20),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(255),
            1
        );

        cv::imshow("MNIST Sample", displayImage);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

void printProgressBar(int epoch, int current, int total, int width = 50) {
    double progress = static_cast<double>(current) / static_cast<double>(total);
    int barWidth = static_cast<int>(progress * width);

    std::cout << "Epoch " << epoch << " [";

    for (int i = 0; i < width; i++) {
        if (i < barWidth) {
            std::cout << "=";
        } else {
            std::cout << " ";
        }
    }

    std::cout << "] " << static_cast<int>(progress * 100.0) << "%\r";
    std::cout.flush();

    if (current >= total) {
        std::cout << std::endl;
    }
}

double evaluateAccuracy(mazeNet& maze, const cv::Mat& X, const cv::Mat& y) {
    CV_Assert(X.rows == y.rows);
    CV_Assert(y.cols == 1);

    cv::Mat output = maze.forward(X);

    int correct = 0;

    for (int i = 0; i < output.rows; i++) {
        int predicted = argmaxRow(output.row(i));
        int actual = static_cast<int>(y.at<double>(i, 0));

        if (predicted == actual) {
            correct++;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(X.rows);
}

int main() {
    helper logLoss;

    loadDataset();

    mazeNet maze(in_size, hide_size, out_size);

    std::cout << "\nStart training loop ...\n" << std::endl;

    int steps = X_train.rows;

    for (int epoch = 0; epoch < total_epochs; epoch++) {
        /*
            Shuffle data at the start of each epoch.
        */

        shuffleDataset(X_train, y_train);

        double epochLoss = 0.0;

        for (int i = 0; i < steps; i += batch_size) {
            int batchEnd = std::min(i + batch_size, steps);

            cv::Mat X_batch = X_train.rowRange(i, batchEnd).clone();
            cv::Mat y_batch = y_train.rowRange(i, batchEnd).clone();

            cv::Mat y_pred = maze.forward(X_batch);

            double batchLoss = logLoss.categoricalCrossEntropy(y_batch, y_pred);
            epochLoss += batchLoss * static_cast<double>(X_batch.rows);

            maze.backward(X_batch, y_batch, y_pred, l_rate);

            printProgressBar(epoch + 1, batchEnd, steps);
        }

        epochLoss /= static_cast<double>(steps);

        double testAccuracy = evaluateAccuracy(maze, X_test, y_test);

        std::cout << "Loss     : " << epochLoss << std::endl;
        std::cout << "Accuracy : " << testAccuracy * 100.0 << "%" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    std::cout << "\nModel has been trained!" << std::endl;

    FileStorage fs("trained_model.yml", FileStorage::WRITE);

    if (fs.isOpened()) {
        fs << "w1" << maze.w1;
        fs << "b1" << maze.b1;
        fs << "w2" << maze.w2;
        fs << "b2" << maze.b2;
        fs.release();

        std::cout << "Model parameters saved to trained_model.yml" << std::endl;
    } else {
        std::cerr << "Failed to open file for saving model parameters." << std::endl;
    }

    return 0;
}
