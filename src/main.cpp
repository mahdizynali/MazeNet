#include "include/config.hpp"
#include "include/utils.hpp"
#include "include/model.hpp"
#include "include/mnist.hpp"

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

    //mnist dataset address
    readUbyte dataset("/home/mahdi/Desktop/MazeNet/dataset/train-images.idx3-ubyte",
                     "/home/mahdi/Desktop/MazeNet/dataset/train-labels.idx1-ubyte");
    Mat X_train = dataset.readImages();
    Mat y_train = dataset.readLabels();    

    int steps = X_train.rows / batch_size;
    double loss = 0;

    // Training loop
    cout<<"Start training loop ...\n";
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
    return 0;
}