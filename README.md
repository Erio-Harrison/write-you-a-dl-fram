# Write you a deep learning framework

# Still working

```bash
sudo apt-get install libbenchmark-dev
```

```bash
g++ tensor_benchmark.cpp -o tensor_benchmark -lbenchmark -pthread -march=native -mavx
```


```bash
g++ -std=c++17 -isystem /usr/include/gtest -pthread tensor_test.cpp -o tensor_test -lgtest -lgtest_main -mavx
```

```bash
g++ -std=c++17 -isystem /usr/include/gtest -pthread tensor_multithreading_test.cpp -o tensor_multithreading_test -lgtest -lgtest_main -mavx
```