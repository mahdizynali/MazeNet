#include "include/config.hpp"
#include "include/utils.hpp"
#include "include/model.hpp"

int main() {

    mazeNet maze(in_size, hide_size, out_size);
    maze.printLayerSize();

    int steps = X_train.rows;
    double loss;

    // Training loop
    for (int epoch = 0; epoch < total_epochs; epoch++) {
        for (int i = 0; i < steps; i += batch_size) {
            int batch_start = i;
            int batch_end = std::min(i + batch_size, steps);  // Ensure not to go beyond the array size

            cv::Mat X_train = X_train.rowRange(batch_start, batch_end);
            cv::Mat y_train = y_train.rowRange(batch_start, batch_end);

            cv::Mat y_pred = maze.forward(X_train);

            loss = utils().categoricalCrossEntropy(y_train, y_pred);

            maze.backward(X_train, y_train, y_pred, l_rate);
        }

        cout << "Epoch " << epoch + 1 << " completed." << endl;
    }

    return 0;
}