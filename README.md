# CUDA Gradient Descent

The goal of the project was to implement a Stochastic Gradient Descent (SGD) algorithm with mini-batches in CUDA, in order to apply concepts of GPU programming seen in course. <br>
The implementation was carried out on Google Colab, the whole code can be viewed in the [attached notebook](./CUDA_Linear_classification.ipynb). An abstraction for matrix objects was used, called fmatrix in the program, to manipulate matrices easilier. <br>
The dataset used to validate the implementation is the [California Housing Data Set Description](https://developers.google.com/machine-learning/crash-course/california-housing-data-description), which is also provided by default in Google Colab notebooks (in the [sample_data](./sample_data) folder). <br>
The SGEMM (Single precision GEneral Matrix Multiply) function of the [cuBLAS library](https://docs.nvidia.com/cuda/cublas/index.html) was used for more efficient matrix multiplication operations. Several experiments were ran to check the impact of hyperparameters (especially batch size, learning rate and number of epochs) and batch shuffling on the accuracy and speed of the algorithm.
