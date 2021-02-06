#include "Image.h"


Image& Image::diagSymm(int diag_type)
{

    if (diag_type == -1){
        cv::Mat_<float> transposedMatrix(width, height);
        for(size_t i{0}; i < height; ++i) {
            for (size_t j{0}; j < width; ++j) {
                transposedMatrix.at<float>(j, i) = matrixImg.at<float>(i, j);
            }
        }
        matrixImg = transposedMatrix;
        height = matrixImg.size().height;
        width = matrixImg.size().width;
        save("../diagonalSymmetryReversed.png");
        return *this;
    }else if (diag_type == 1){
        cv::Mat_<float> transposedMatrix(width, height);
        for(size_t i{0}; i < height; ++i) {
            for (size_t j{0}; j < width; ++j) {
                transposedMatrix.at<float>(j, i) = matrixImg.at<float>(height - i, width - j);
            }
        }
        matrixImg = transposedMatrix;
        height = matrixImg.size().height;
        width = matrixImg.size().width;
        save("../diagonalSymmetryDirect.png");
        return *this;
    }else{
        throw std::runtime_error("");
    }

}


// Symmetry transform along the y axis
Image& Image::symmTransformY()
{
    if (!matrixImg.empty()){
        cv::Mat_<float> resultImg(height,width);
        for (size_t i{0}; i < height; i++){
            for (size_t j{0}; j < width; j++){
                resultImg.at<float>(i,width-1-j) = matrixImg.at<float>(i,j);
            }
        }
        matrixImg = resultImg;
        save("../symmetry_transform_Y.png");
        return *this;
    } else{
        throw std::runtime_error("No image data.");
    }
}


// Symmetry transform along the x axis
Image& Image::symmTransformX()
{
    if (!matrixImg.empty()){
        cv::Mat_<float> resultImg(height,width);
        for (size_t i{0}; i < height; i++){
            for (size_t j{0}; j < width; j++){
                resultImg.at<float>(height-1-i,j) = matrixImg.at<float>(i,j);
            }
        }
        matrixImg = resultImg;
        save("../symmetry_transform_X.png");
        return *this;
    } else{
        throw std::runtime_error("No image data.");
    }
}






