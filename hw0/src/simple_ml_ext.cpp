#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void normalize_(float* input, int n, int k) {
    // inplace normal_
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            input[i * k + j] = std::exp(input[i * k + j]);
            sum += input[i * k + j];
        }
        for (int j = 0; j < k; ++j) {
            input[i * k + j] = input[i * k + j] / (sum);
        }
    }
}

void transpose(const float* A, float* B, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (int i = 0; i < m; i += batch) {
        float* X_T = new float[n * batch];
        float* Z = new float[batch * k];
        float* deta = new float[n * k];
        const float* X_mini = new float[n * batch];
        // const unsigned char *y_mini = new unsigned char[batch];
        X_mini = &X[i * n];
        // y_mini = &y[i];
        // X += batch * n; # 常量指针不能自加
        // y += batch;
        // X += i * n;
        // y += i;
        transpose(X_mini, X_T, batch, n);
        matmul(X_mini, theta, Z, batch, n, k); // batch * n @ n * k  矩阵乘完 nan了 为啥
        // transpose(X, X_T, batch, n);
        // matmul(X, theta, Z, batch, n, k); // batch * n @ n * k  矩阵乘完 nan了 为啥
        
        normalize_(Z, batch, k); // batch * k
        for (int bid = 0; bid < batch; ++bid) {
            // Z[bid * k + y_mini[bid]] -= 1.0;
            Z[bid * k + y[bid + i]] -= 1.0;
        }
        matmul(X_T, Z, deta, n, batch, k); // n * batch @  batch * k,
        
        for (int theta_ = 0; theta_ < n * k; ++theta_) {
            theta[theta_] -= lr / batch * deta[theta_];
        }
        delete[] X_T;
        delete[] Z;
        delete[] deta;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
