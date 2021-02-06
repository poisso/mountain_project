#include"Image.h"
#include"utils.h"

// A method which calculates the pressure coefficients for a circular model using exp(-r)
cv::Mat Image::pressureCoeffsExp(const std::string &strength, const int &centerY, const int &centerX, const float &speed)
{
	if (!matrixImg.empty()){
		float rMax;
		if (abs(height - centerY) >= abs(width - centerX)){
			rMax = abs(width - centerX);	
		} else{
			rMax = abs(height - centerY);	
		}
		cv::Mat coeffsMat(height,width,CV_32F,cv::Scalar(0));
		for (int i{0}; i < height; i++){
			for (int j{0}; j < width; j++){
				double r{sqrt(pow(centerY - i,2.) + pow(centerX - j,2.))};
				coeffsMat.at<float>(i,j) = exp((-speed*r)/rMax);
			}
		}
		if (strength == "strong" || strength == "s"){	
			coeffsMat = 1. - coeffsMat;
		} else if (strength == "weak" || strength == "w"){
			;
		} else{
			throw std::runtime_error("Must select either strong or weak pressure!");
		}
		return coeffsMat;
	} else{
		throw std::runtime_error("No image data.");
	}
}

// A method which calculates the pressure coefficients for a polynomial model using the formula: f(x) = 1. - x^(speed)
cv::Mat Image::pressureCoeffsPoly(const std::string &strength, const int &centerY, const int &centerX, const float &speed)
{
	if (!matrixImg.empty()){
		float rMax;
		if (abs(height - centerY) >= abs(width - centerX)){
			rMax = abs(width - centerX);	
		} else{
			rMax = abs(height - centerY);	
		}
		cv::Mat coeffsMat(height,width,CV_32F,cv::Scalar(0));
		for (int i{0}; i < height; i++){
			for (int j{0}; j < width; j++){
				double r{sqrt(pow(centerY - i,2.) + pow(centerX - j,2.))};
				coeffsMat.at<float>(i,j) = 1. - pow(r/rMax,speed);
			}
		}	
		if (strength == "strong" || strength == "s"){	
			coeffsMat = 1. - coeffsMat;
		} else if (strength == "weak" || strength == "w"){
			;
		} else{
			throw std::runtime_error("Must select either strong or weak pressure!");
		}
		return coeffsMat;
	} else{
		throw std::runtime_error("No image data.");
	}
}

// A method which applies the pressure coefficients matrix on an image
Image& Image::pressureTransform(const std::string &strength, const std::string &transformType, const int &centerY, const int &centerX, const float &speed)
{
	if (!matrixImg.empty()){
		cv::Mat coeffsMat;
		if (transformType == "exponential" || transformType == "exp"){ 
			coeffsMat = this->pressureCoeffsExp(strength,centerY,centerX,speed);
		} else if(transformType == "polynomial" || transformType == "poly"){
			coeffsMat = this->pressureCoeffsPoly(strength,centerY,centerX,speed);
		} else{
			throw std::runtime_error("No transform type given.");
		}
		if (strength == "weak" || strength == "w"){		
			this->invertColors();
		}
		for (size_t i{0}; i < height; i++){
			for (size_t j{0}; j < width; j++){	
				matrixImg.at<float>(i,j) = coeffsMat.at<float>(i,j)*matrixImg.at<float>(i,j);
			}
		}
		if (strength == "weak" || strength == "w"){		
			this->invertColors();
		}
		cv::imwrite("../pressure_transform.png",u::decode(matrixImg));
		return *this;
	} else{
		throw std::runtime_error("No image data.");
	}
}


// Anisotropic pressure 
Image& Image::anisotropicLocalPressure(double ellipse_width, double ellipse_height, int x, int y){
    // We create the temporary matrix in which we will store the result
    cv::Mat_<float> tmp(height, width);
    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){
            tmp.at<float>(i, j) = 1;
        }
    }
    cv::Point centre(x, y);
    // Declaration of the rotation matrix which will be changed for each pixel


    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){
            // We change of basis

            int Y = i - centre.y;
            int X = j - centre.x;
            double ellipse_scale_coef = (X*X)/(ellipse_width*ellipse_width) + (Y*Y)/(ellipse_height*ellipse_height);
            double new_width_carre  = ellipse_width*ellipse_width*ellipse_scale_coef;

            // We must first calcultate the angle of rotation depending on the distance to the focis
            double distance_to_focis = 2*sqrt(new_width_carre);
            //for silly pressure:
            double intensity_coef = 2 - exp(-pow(distance_to_focis, 2)/(2*pow(ellipse_width, 2)));
            
            //for strong pressure:
            //double intensity_coef = 1 - exp(-pow(distance_to_focis, 2)/(2*pow(ellipse_width, 2)));
            
            // We make sure the new pixel is in the boundaries of the image
            tmp.at<float>(i, j) = matrixImg.at<float>(i, j)*intensity_coef;

        }
    }
    matrixImg = u::motionInterpolation(tmp, 0.99);
    return *this;
}


// Simple Algorithm for finding a low pressure mask for an approximate fingerprint shape 
cv::Mat Image::extractWeakPressureMask()
{
	Image weak_finger("../data/weak_finger.png");
	weak_finger.invertColors();
	this->superpose(weak_finger);
	this->convolve(Average_blurring,11);
	this->invertColors();
	this->contrastThresholding(0.99);
	this->removeBorder("b",8);
	cv::Mat tmp(height,width,CV_32F,cv::Scalar(0));
	matrixImg.copyTo(tmp);
	return tmp;
}

// DOESN'T WORK (yet)
// A method that returns a matri of pressure coeffs
cv::Mat Image::anisotropicWeakPressureCoeffs(const cv::Point &center, const double &rMax, const double &rMin, const double &speed)
{
	cv::Mat pressureCoeffs(height,width,CV_32F,cv::Scalar(0));
	for (int i{0}; i < height; i++){
		for (int j{0}; j < width; j++){
			int y{center.y - i};
			int x{j - center.x};
			double theta;
			if (x == 0){
				if (y > 0){
					theta = M_PI/2;
				} else if (y < 0){
					theta = 3*M_PI/2;
				} else{
					theta = 0.;
				}	
			} else{
				theta = atan(y/x);
			}
			double r{sqrt((pow(rMax,2)*pow(rMin,2))/(pow(rMin,2)*pow(cos(theta),2) + pow(rMax,2)*pow(sin(theta),2)))};
			std::cout << r/rMax << "\n";
			pressureCoeffs.at<float>(i,j) = 1. - pow(r/rMax,speed);
		}
	}
	cv::imshow("pressure coeffs",pressureCoeffs);	
	cv::waitKey(0);
	return pressureCoeffs;
}

// A method which applies weak anisotropic pressure to the image and uses the mask from weak_finger.png to create a realistic recreatin of weak_finger.png
Image& Image::anisotropicWeakPressureTransform(const float &speed)
{
	cv::Mat lowPressureMask = this->extractWeakPressureMask();	
	cv::Point center = u::barycentre(matrixImg);

	int iMin = height - 1; int iMax = 0; 
	int jMin = width - 1; int jMax = 0;
	for(int i{0};i < height;i++){
		for (int j{0}; j < width; j++){
			if (matrixImg.at<float>(i,j) == 0.){
				if (i < iMin) iMin = i;
				if (i > iMax) iMax = i;
				if (j < jMin) jMin = j;
				if (j > jMax) jMax = j;				 
			}
		}
	}
	this->load();
	this->anisotropicLocalPressure((jMax - jMin)/2 + 15,(iMax - iMin)/2 + 7,center.x,center.y);
	this->invertColors();
	for (int i{0}; i < height; i++){
		for (int j{0}; j < width; j++){
			matrixImg.at<float>(i,j) = matrixImg.at<float>(i,j)*(1. - lowPressureMask.at<float>(i,j));
		}
	}
	this->invertColors();
	return *this;
}

