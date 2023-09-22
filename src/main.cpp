#include "include/config.hpp"
#include "include/utils.hpp"
#include "include/model.hpp"
#include "include/mnist.hpp"

int main() {

    helper logLoss;

    mazeNet maze(in_size, hide_size, out_size);
    // maze.printLayerSize();

    //mnist dataset address
    readUbyte dataset("/home/mahdi/Desktop/MazeNet/dataset/train-images.idx3-ubyte",
                     "/home/mahdi/Desktop/MazeNet/dataset/train-labels.idx1-ubyte");
    Mat X_train = dataset.readImages();
    Mat y_train = dataset.readLabels();    

    int steps = X_train.rows;
    double total_loss = 0.0;

    // Training loop
    
    for (int epoch = 0; epoch < total_epochs; epoch++) {
        for (int i = 0; i < steps; i += batch_size) {
            int batch_start = i;
            int batch_end = std::min(i + batch_size, steps);  // Ensure not to go beyond the array size
            cv::Mat X_batch = X_train.rowRange(batch_start, batch_end);
            cv::Mat y_batch = y_train.rowRange(batch_start, batch_end);
            
            cv::Mat y_pred = maze.forward(X_batch);
            
            total_loss += logLoss.categoricalCrossEntropy(y_batch, y_pred);
            
            maze.backward(X_batch, y_batch, y_pred, l_rate);         
        }
        double average_loss = total_loss / (steps / batch_size);
        cout << "Epoch " << epoch + 1 << " completed. Average Loss: " << average_loss << endl;
    }
    return 0;
}