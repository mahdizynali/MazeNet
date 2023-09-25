# MazeNet
Mnist multi perceptron neural-network training from scratch with c++ and opencv matrix. \
in this repository there are two folder that includes folder contains header files and src contains cpp programs.
## How to train
First clone repository via this command :
```
git clone https://github.com/mahdizynali/MazeNet.git
```
after that you have to extract dataset from compress file (dataset.tar.xz).
tar.gz is being support by linux & mac systems. \
you would change config.hpp header in order to set your hyperparameter as you need.

```
# define l_rate 0.005
# define total_epochs 10
# define batch_size 64
```
right now inside MazeNet folder follow these commands to build and run training loop :
```
mkdir build && cd build
```
```
make
```
Or try this for a bit faster :
```
make -j`nproc`
```
finally run the program :
```
./maze
```
after training , neuron weights will recive and save into a yml file besides maze program.
