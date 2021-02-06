#include "Image.h"
#include "utils.h"
#include <chrono>
#include <ctime>

Image &Image::convolve(FilterType filterType, int size, bool deconvolve) {
    if(filterType == FilterType::SV){
        filterType = FilterType::Gaussian_blurring;
        cv::Mat filter = u::setFilter(filterType, size, 1);
        float sigma{0.0f};
        cv::Point bc(u::barycentre(matrixImg));
        std::cout << "BAry = " << bc.x << " x " << bc.y << std::endl;
        //some test centers
        std::vector<cv::Point> centres;
        cv::Point pos1(60,100);
        cv::Point pos2(100,60);
        cv::Point pos3(130,150);
        cv::Point pos4(50,190);
        cv::Point pos5(180,90);
        cv::Point pos6(150,150);
        cv::Point pos7(140,180);
        centres.push_back(pos1); centres.push_back(pos2); centres.push_back(pos3);
        centres.push_back(pos4); centres.push_back(pos5); centres.push_back(pos6);
        centres.push_back(pos7);
        std::vector<Window> windows;
        u::setWindows(windows, centres, matrixImg);


        for (auto & window : windows) {
            sigma = u::getSigma(window, bc,  width, height, false);
            filter = u::setFilter(filterType, size, sigma);
                window.window = u::expConvolveFrequential(window.window, filter, deconvolve);
        }
        u::smartEmplaceWindows(windows, matrixImg);
        return *this;
    }else{
        cv::Mat filter = u::setFilter(filterType, size,  1.);
        if(size < 15 && !deconvolve){
            matrixImg = u::convolveSpatial(matrixImg, filter);
        }
        else{
            matrixImg = u::convolveFrequential(matrixImg, filter, deconvolve);
        }
        std::cout << "Size of image is " << matrixImg.size().width << " x " << matrixImg.size().height << std::endl;
        return *this;
    }
}


void Image::convolveOPENCV(int kernelSize, const std::string& type) {
    auto start = std::chrono::system_clock::now();
    cv::Mat bluredImage(height, width, CV_32F, cv::Scalar(0.));

    if(type == "gaussian"){
        cv::Mat kernel = cv::getGaussianKernel(kernelSize*kernelSize,1);
        std::cout << "OPENCV KERNEL " << kernel << std::endl;
        cv::filter2D(matrixImg, bluredImage, -1 , kernel, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT );
    }
    if(type == "blurring"){
        cv::Mat kernel = cv::Mat::ones( kernelSize, kernelSize, CV_32F )/ (float)(kernelSize*kernelSize);
        cv::filter2D(matrixImg, bluredImage, -1 , kernel, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT );
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "OpenCV convolution finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    imshow( "Blurred with openCV convolution.", bluredImage);
    matrixImg = bluredImage;
    cv::waitKey ( 0 );
}

// A method for removing borders
Image& Image::removeBorder(const std::string &BorW, const int &thickness)
{
	float value;
	if (BorW == "black" || BorW == "b"){
		value = 1.;
	} else if(BorW == "white" || BorW == "w"){
		value = 0.;
	} else{
		throw std::runtime_error("Must enter whether the border is black or white!");
	}
	for(int i{0}; i < thickness; i++){
		for(int j{0}; j < width; j++){
			matrixImg.at<float>(i,j) = value;
		}
	}
	for(int i{height - thickness}; i < height; i++){
		for(int j{0}; j < width; j++){
			matrixImg.at<float>(i,j) = value;
		}
	}
	for(int i{0}; i < height; i++){
		for(int j{0}; j < thickness; j++){
			matrixImg.at<float>(i,j) = value;
		}
	}
	for(int i{0}; i < height; i++){
		for(int j{width - thickness}; j < width; j++){
			matrixImg.at<float>(i,j) = value;
		}
	}
	return *this;
}

