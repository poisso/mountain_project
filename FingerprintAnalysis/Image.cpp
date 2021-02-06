#include "Image.h"


// Parametrized constructor
Image::Image(const std::string &src) {
	this->src = src;
	if(load() != 0){
        throw std::runtime_error("No image data.");
    }
}

// Destructor
Image::~Image() {}

// A method which loads an image from a given path.
int Image::load() {
	cv::Mat img = cv::imread(src, cv::IMREAD_GRAYSCALE);
	if(img.empty()){
		std::cout << "Could not read the image: " << src << std::endl;
		return 1;
	}
	width = img.size().width;
	height = img.size().height;
	std::cout << "Size of image is " << width << " x " << height << std::endl;

	// Mapping to float values between 0 and 1 and storing them in matrixImg
	img.convertTo(matrixImg, CV_32F);
	matrixImg /= 255.;
	return 0;
}

// A method which visualizes an image while first converting it to CV_8UC1 standard
void Image::show() {
	cv::Mat decoded = u::decode(matrixImg);
	const std::string window_name("Image");
   	cv::namedWindow(window_name);
	cv::imshow(window_name, decoded);
	cv::waitKey(0);
}

// A method which saves an image
void Image::save(const std::string &src) {
	cv::imwrite(src, u::decode(matrixImg));
}

// FIX
void Image::findMinMax() {
	minIntensity = 255;
	maxIntensity = 0;
	for (int i{0}; i < width; i++) {
		for(int j{0}; j < height; j++){
			auto color = matrixImg.at<float>(i, j);
			if(color < minIntensity){
				minIntensity = color;
			}
			if(color > maxIntensity){
				maxIntensity = color;
			}	
		}
	}
	std::cout << "Maximum intensity is " << maxIntensity << ", minimum: " << minIntensity << std::endl;
}

// FIX
Image& Image::drawRandomSquares() {
	//For black square
	srand (time(NULL));
	int xMin = rand() % int(width);
	int xMax = rand() % int(width - xMin) + xMin;
	int yMin = rand() % int(height);
	int yMax = rand() % int(height - yMin) + yMin;


    for (int x{xMin}; x < xMax; ++x) {
		for(int y{yMin}; y < yMax; ++y){
			matrixImg.at<float>(x, y) = 0.;
		}
	}

	//For white square
    xMin = rand() % int(width);
	xMax = rand() % int(width - xMin) + xMin;
	yMin = rand() % int(height);
	yMax = rand() % int(height - yMin) + yMin;

	for (int x{xMin}; x < xMax; ++x) {
		for(int y{yMin}; y < yMax; ++y){
			matrixImg.at<float>(x, y) = 1.;
		}
	}
	return *this;
}

// A method which prints the numerical values of each pixel of the image matrix
void Image::printVals()
{
	for (size_t i{0}; i<height; i++){
		for (size_t j{0}; j<width; j++){
			std::cout << matrixImg.at<float>(i,j) << " ";		
		}
		std::cout << std::endl;
	}
}

// A method which inverts the colors of an image
Image& Image::invertColors()
{
	if (!matrixImg.empty()){
		matrixImg = 1. - matrixImg;
		return *this;
	} else{
		throw std::runtime_error("No image data.");
	}
}

// Thresholding contrast transformation. If a pixel float intensity(between 0 and 1) is below the threshold, it will be put to 0, if it is above it, it will be put to 1.
Image& Image::contrastThresholding(const float &level)
{
	if (!matrixImg.empty()){
		if (level >= 0. && level <= 1.){
			for (size_t i{0}; i < height; i++){
				for (size_t j{0}; j < width; j++){
					if (matrixImg.at<float>(i,j) < level){
						matrixImg.at<float>(i,j) = 0.;
					} else{
						matrixImg.at<float>(i,j) = 1.;
					}	
				}
			}
			return *this;
		} else{
			throw std::runtime_error("The level of thresholding must be between 0.0 and 1.0");
		}
	} else{
		throw std::runtime_error("No image data.");
	}
}

/*!
 * A method for superposing two images together
 * \param[in] const Image & the image to superpose with
*/
Image &Image::superpose(const Image &otherImage)
{
	for (size_t i{0}; i<height; i++){
		for (size_t j{0}; j<width; j++){
			matrixImg.at<float>(i,j) *= otherImage.matrixImg.at<float>(i,j); 
		}
	}
	return *this;
}













