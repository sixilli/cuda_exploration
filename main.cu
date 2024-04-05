#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

__global__
void colorToGrayScale(int height, int width, int channels, unsigned char* d_img_in, unsigned char* d_img_out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int offset = row * width + col;

        int rbg_offset = offset * channels;

        unsigned char r = d_img_in[rbg_offset    ];
        unsigned char g = d_img_in[rbg_offset + 1];
        unsigned char b = d_img_in[rbg_offset + 2];

        d_img_out[offset] = r * 0.21f + g * 0.71f + b * 0.07f;
    }
}

int main() {
    std::string image_path = samples::findFile("teste.jpg");
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty()) {
        std::cout << "Could not read image" << std::endl;
        return 1;
    }

    int height = img.size().height;
    int width = img.size().width;
    int channels = img.channels();

    std::cout << "channels: " << channels << " width: " << width << " height: " << height << std::endl;

    cvtColor(img, img, COLOR_BGR2RGB, 0);

    int rows, cols;
    if (img.isContinuous()) {
        rows = 1;
        cols = height * width * channels;
    } else {
        rows = height;
        cols = width;
    }

    std::cout << "building img_vals" << std::endl;
    int cuda_img_size = height * width * channels;
    std::cout << "cuda_size: " << cuda_img_size << std::endl;
    unsigned char* img_vals = new unsigned char[cuda_img_size];
    uchar* p;
    std::cout << "hi" << std::endl;
    for (int row = 0; row < rows; row++) {
        p = img.ptr<uchar>(row);
        for (int col = 0; col < cols; col++) {
            int offset = row * width + col;
            img_vals[col] = p[col];
        }
    }

    std::cout << "allocating cuda stuff" << std::endl;
    unsigned char *d_img_in; cudaMallocManaged(&d_img_in, cuda_img_size * sizeof(unsigned char));
    unsigned char *d_img_out; cudaMallocManaged(&d_img_out, height * width * sizeof(unsigned char));

    std::cout << "doing copy stuff into cuda stuff" << std::endl;
    std::copy(&img_vals[0], &img_vals[0] + cuda_img_size, d_img_in);

    std::cout << "kernal funsies" << std::endl;
    dim3 dimGrid(ceil (height/16.0), ceil(width/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayScale<<<dimGrid, dimBlock>>>(height, width, channels, (unsigned char *)d_img_in, (unsigned char *)d_img_out);

    cudaDeviceSynchronize();

    std::cout << "building out image" << std::endl;
    Mat out_img(height, width, CV_8UC1, Scalar(0, 0, 0));

    rows = 1;
    cols = height * width;
    for (int row = 0; row < rows; row++) {
        p = out_img.ptr<uchar>(row);
        for (int col = 0; col < cols; col++) {
            int offset = row * width + col;
            p[col] = d_img_out[col];
        }
    }

    imwrite("cuda_dump/test2.jpg", out_img);


    std::cout << "saved image" << std::endl;
    return 0;
}
