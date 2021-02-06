/*!
 *  \brief File contains a lot of methods which are useful in image processing
 *  but can't be assumed as class Image methods because they are more general.
 *  \author Alisa Barkar, Andrew McDonald, Nikolay Tverdokhleb, Veselin Doychinov
 *  \version 1.0
 *  \date 2021 January
 *
*/
#pragma once
#include <iostream>
#include <string>
#include <cmath>
#include "opencv2/opencv.hpp"
/**
 * @brief Names of supported filters.
 */
enum FilterType{
        Average_blurring,
        Gaussian_blurring,
        Sharpening_Sobol_X,
        Sharpening_Sobol_Y,
        SV,
        Test,
        SV_Gaussian_blurring,
        SV_Custom_blurring,
        SV_Inverse_Custom_blurring
};

/**
 * @brief Structure to store window which is used in filtering.
 */
struct Window{
    cv::Mat window;
    cv::Point centre;
    int sizeX = 80, sizeY = 80;
};

/**
 * @brief Utility functions.
 */
namespace u {
    cv::Mat_<float> multiplication(cv::Mat_<float> A, cv::Mat_<float> B);///<Performs matrix multiplication

    cv::Point barycentre(cv::Mat img);///< General utility functions for images
    cv::Mat decode(cv::Mat img);///<Decoding function to show and write images

    //Motions
    cv::Mat_<float> pixelRotation(int x, int y, cv::Mat_<float> rotation_matrix);///<Utility function for the part about motions
    float meanAroundPixel(cv::Mat_<float> img, int i, int j);///<Interpolation functions
    cv::Mat_<float> motionInterpolation(cv::Mat_<float> img, float threshold);///<

    //Filtering
    cv::Mat convolveSpatial(const cv::Mat &image, const cv::Mat &filter);///<Performs spatial convolution
    cv::Mat convolveFrequential(const cv::Mat &image, const cv::Mat &filter, bool deconvolve);///<Performs frequential convolution with zero-padding
    cv::Mat expConvolveFrequential(const cv::Mat &image, const cv::Mat &filter, bool deconvolve);///<Performs frequential convolution without zero-padding
    cv::Mat setFilter(FilterType filterType, int size, float sigma = 0);///<Sets a matrix of spatially invariant filter
    cv::Mat setSVFilter(FilterType filterType, int size, float sigma, cv::Point barycente, Window window);///<Sets a matrix of spatially variant filter
    float getSigma(Window window, const cv::Point &barycentre, int imgSizeX, int imgSizeY, bool inverse);///<Calculates sigma for Gaussian filter
    static void divSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB);///<Divides spectrums of CCS-packed matrices
    void mul(const cv::Mat &M1Real, const cv::Mat &M1Complex, const cv::Mat &M2Real, const cv::Mat &M2Complex,
             cv::Mat &ResReal, cv::Mat &ResComplex);///<Multiplies spectrums of two complex matrices
    void div(const cv::Mat &M1Real, const cv::Mat &M1Complex, const cv::Mat &M2Real, const cv::Mat &M2Complex,
             cv::Mat &ResReal, cv::Mat &ResComplex);///<Divides spectrums of two complex matrices
    void splitFilter(const cv::Mat &filter, cv::Mat &resFilter);///<Splits filter to fit in matrix of image size

    //Windowing
    void setWindows(std::vector<Window> &windows, const std::vector<cv::Point> &centres, const cv::Mat &image);///<Sets array of windows corresponding to it's centres and sizes
    cv::Mat getWindow(const cv::Point &position, int sizeX, int sizeY, const cv::Mat &image, cv::Mat &showWindows);///<Fills window with corresponding image's values
    void emplaceWindows(std::vector<Window> &windows, cv::Mat &image);///<Emplace window back in image
    void smartEmplaceWindows(std::vector<Window> &windows, cv::Mat &image);///<Emplace window back in image and check if windows are intercrossing
    bool isWindowsIntercross(const Window &window1, const Window &window2);///<Checks if two windows intercross
    bool isPointInWindow(const cv::Point &point, const Window &window);///<Checks if the point is inside the window
    std::vector<int> indexWindowPoint(cv::Point &point, const std::vector<Window> &windows);///<Creates an array with indexes of windows containing point
    std::vector<float>
    getRads(const cv::Point &point, const std::vector<Window> &windows, const std::vector<int> &indexes);///<Calculates the distance between centre of window and point's coordinates
    float getValueOfWindow(const cv::Point &point, Window &window);///<Returns value of window in point in image's coordinates

    // Finite difference scheme
    cv::Mat_<float> horizontalDerivative(cv::Mat_<float> img);///<
    cv::Mat_<float> verticalDerivative(cv::Mat_<float> img);///<

    // Partial derivative of loss function
    double nablaPx(cv::Mat_<float> img, cv::Mat_<float> other);///<
    double nablaPy(cv::Mat_<float> img, cv::Mat_<float> other);///<

    // Calculate the translation needed to match barycentres
    cv::Point matchBarycentres(cv::Mat_<float> img, cv::Mat_<float> other);///<
}