/*!
 *  \brief This class allows to perform things with images.
 *  \author Alisa Barkar, Andrew McDonald, Nikolay Tverdokhleb, Veselin Doychinov
 *  \version 1.0
 *  \date 2021 January
 *
*/
#pragma once
#include <iostream>
#include <string>
#include <cmath>
#include "utils.h"
#include "opencv2/opencv.hpp"
class Image {
protected:
    std::string src;
    int width;
    int height;
    int minIntensity;
    int maxIntensity;
    cv::Mat_<float> matrixImg;
private:
    int load();///< Loads the image from a file
public:
    // Constructor and destructor, defined in 'Image.cpp'
    Image(const std::string &src);

    // Motion model methods
    Image& similarity(double theta, double scale, double tx, double ty);
    Image& isotropicLocalSimilarity(double theta, double radius, int x, int y);
    Image& anisotropicLocalSimilarity(double theta, double ellipse_width, double ellipse_height, int x, int y);

    // Destructor
    ~Image();

    // Basic methods, defined in 'Image.cpp'
    virtual void show();
    void save(const std::string &src);
    void findMinMax();
    void update();
    void printVals();
    Image& drawRandomSquares();

    // Basic transformations, defined in 'Image.cpp'
    Image &invertColors();
    Image &contrastThresholding(const float &level);
    Image &superpose(const Image &otherImage);

    // Symmetry transformations, defined in 'SymmetryTransformations.cpp'
    Image &diagSymm(int diag_type);
    Image &symmTransformY();
    Image &symmTransformX();

    // Pressure methods and transformations, defined in 'Pressure.cpp'
    // isotropic
    cv::Mat pressureCoeffsExp(const std::string &strength, const int &centerY, const int &centerX, const float &speed);
    cv::Mat pressureCoeffsPoly(const std::string &strength, const int &centerY, const int &centerX, const float &speed);
    Image &pressureTransform(const std::string &strength, const std::string &transformType, const int &centerY, const int &centerX, const float &speed);
    // anisotropic
    cv::Mat extractWeakPressureMask();    
    cv::Mat anisotropicWeakPressureCoeffs(const cv::Point &center, const double &rMax, const double &rMin, const double &speed);
    Image& anisotropicWeakPressureTransform(const float &speed);
    Image& anisotropicLocalPressure(double ellipse_width, double ellipse_height, int x, int y);

    //Filtering functions
    Image &convolve(FilterType filterType, int size, bool deconvolve = false);
    Image &removeBorder(const std::string &BorW, const int &thickness);
    void convolveOPENCV(int kernelSize, const std::string& type);

    // Registrations functions
    Image &plotLoss(Image &other);
    int lossFunction(Image &other);
    cv::Point gradientDescent(Image &other, double alpha, double epsilon);
    void uselessPlot(Image &other);
};

