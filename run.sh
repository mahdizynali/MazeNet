clear

if [ -d "dataset" ]; then
    echo "Datasets successfully find."
else
    tar -xvf dataset.tar.xz
    echo "Datasets successfully made."
fi

if [ -d "build" ]; then
    cd build
else
    mkdir build && cd build
fi

if cmake .. && make -j"$(nproc)"; then
    ./maze
else
    echo "Compilation failed. Check the build process for errors."
fi