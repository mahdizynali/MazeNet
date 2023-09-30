#include "include/config.hpp"
#include "include/utils.hpp"
#include "include/model.hpp"
#include "include/mnist.hpp"

cv::Mat X_train;
cv::Mat y_train;
cv::Mat X_test;
cv::Mat y_test;

void loadDataset () {

    //mnist dataset address
    readUbyte dataset("/home/maximum/Desktop/MazeNet/dataset/train-images.idx3-ubyte",
                    "/home/maximum/Desktop/MazeNet/dataset/train-labels.idx1-ubyte"); 
    cv::Mat X_train_full = dataset.readImages();
    cv::Mat y_train_full = dataset.readLabels();

    // Define the percentage of data to use for training (e.g., 80%)
    double train_percent = 0.8;
    int num_samples = X_train_full.rows;
    int split_index = static_cast<int>(train_percent * num_samples);

    // Split the data into training and test sets
    X_train = X_train_full.rowRange(0, split_index);
    y_train = y_train_full.rowRange(0, split_index);
    X_test = X_train_full.rowRange(split_index, num_samples);
    y_test = y_train_full.rowRange(split_index, num_samples);
}

void displayRandom() {
    std::vector<int> sampleIndices;
        for (int i = 0; i < 5; ++i) {
            int randomIndex = std::rand() % X_train.rows;
            sampleIndices.push_back(randomIndex);
        }

        // Display the selected samples
        for (int i = 0; i < sampleIndices.size(); ++i) {
            int index = sampleIndices[i];
            cv::Mat image = X_train.row(index).reshape(0, 28); // Reshape to 28x28
            cv::String label = std::to_string(static_cast<int>(y_train.at<float>(index, 0)));

            // Create a window and display the image with its label
            cv::imshow("MNIST Sample", image);
            cv::putText(image, label, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            cv::imshow("MNIST Sample with Label", image);

            // Wait for a key press and close the window when a key is pressed
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
}

void printProgressBar(int epoch, int current, int total, int width = 50) {
    float progress = static_cast<float>(current) / total;
    int barWidth = static_cast<int>(progress * width);

    std::cout << "Epoch " << epoch <<" [";
    for (int i = 0; i < width; ++i) {
        if (i < barWidth) {
            std::cout << "=";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << int(progress * 100.0) << "%\r";
    std::cout.flush();

    if (current == total) {
        std::cout << std::endl;
    }
}

int main() {

    helper logLoss;

    mazeNet maze(in_size, hide_size, out_size);
    // maze.printLayerSize();
    loadDataset();
    // displayRandom();

    int steps = X_train.rows / batch_size;
    double loss = 0;

    // Training loop
    cout<<"\nStart training loop ...\n";
    for (int epoch = 0; epoch < total_epochs; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < steps; i += batch_size) {
            int batch_start = i;
            int batch_end = std::min(i + batch_size, steps);  // Ensure not to go beyond the array size
            cv::Mat X_batch = X_train.rowRange(batch_start, batch_end);
            cv::Mat y_batch = y_train.rowRange(batch_start, batch_end);
            
            cv::Mat y_pred = maze.forward(X_batch);
            
            loss = logLoss.categoricalCrossEntropy(y_batch, y_pred);

            maze.backward(X_batch, y_batch, y_pred, l_rate);    
            printProgressBar(epoch, i + 1, steps);   
        }
        cout << "\nLoss: " << loss << endl;
        cout << "\n\n\n";
    }

    cout<<"model has been trained !\n";
    // Mat test_output = maze.forward(X_test);
    // Mat predictions;
    // cv::reduce(test_output, predictions, 1, cv::REDUCE_MAX); // Get index of the max value along rows

    // // Calculate accuracy
    // int correct = 0;
    // for (int i = 0; i < predictions.rows; ++i) {
    //     if (static_cast<int>(y_test.at<float>(i, 0)) == predictions.at<int>(i, 0)) {
    //         correct++;
    //     }
    // }
    // double accuracy = static_cast<double>(correct) / static_cast<double>(X_test.rows);
    // cout << "Accuracy: " << accuracy << endl;

    // Save the trained model's parameters
    FileStorage fs("trained_model.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "w1" << maze.w1;
        fs << "b1" << maze.b1;
        fs << "w2" << maze.w2;
        fs << "b2" << maze.b2;
        fs.release();
    } else {
        cerr << "Failed to open file for saving model parameters." << endl;
    }

    return 0;
}