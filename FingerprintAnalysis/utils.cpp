#include "utils.h"

cv::Mat_<float> u::multiplication(cv::Mat_<float> A, cv::Mat_<float> B){
    if (A.size().width != B.size().height){
        std::cout << "could not perform the mult, not right dim" << std::endl;
        return A;
    }
    cv::Mat_<float> C(A.size().height, B.size().width);
    for (int i{0}; i < C.size().height; i++){
        for (int j{0}; j < C.size().width; j++){
            float coef = 0;
            for (int k{0}; k < A.size().width; k++){
                coef += A.at<float>(i, k)*B.at<float>(k, j);
            }
            C.at<float>(i, j) = coef;
        }
    }
    return C;
}


cv::Point u::barycentre(cv::Mat img){
    float sum_pixel_intensity = 0;
    float x = 0;
    float y = 0;
    for (int i{0}; i < img.size().height; i++){
        for (int j{0}; j < img.size().width; j++){
            img.at<float>(i, j) = 1 - img.at<float>(i, j);
            float value = img.at<float>(i, j);
            y += value*i;
            x += value*j;
            sum_pixel_intensity += value;
            img.at<float>(i, j) = 1 - img.at<float>(i, j);
        }
    }
    cv::Point P(x/sum_pixel_intensity, y/sum_pixel_intensity);
    return P;
}

cv::Mat u::decode(cv::Mat img){
	cv::Mat tmp(img.rows, img.cols, CV_32F);
    img.copyTo(tmp);
	tmp *= 255.;
	cv::Mat decoded;
	tmp.convertTo(decoded, CV_8UC1);
	return decoded;
}

cv::Mat_<float> u::pixelRotation(int x, int y, cv::Mat_<float> rotation_matrix){
    cv::Mat_<float> old_coordinates(3, 1);
    old_coordinates.at<float>(0, 0) = x;
    old_coordinates.at<float>(1, 0) = y;
    // this 1 coef is for the translation
    old_coordinates.at<float>(2, 0) = 1;
    cv::Mat_<float> res = u::multiplication(rotation_matrix, old_coordinates);
    return res;
}

float u::meanAroundPixel(cv::Mat_<float> img, int i, int j){
    return (img.at<float>(i + 1, j) + img.at<float>(i + 1, j + 1) + img.at<float>(i, j + 1) + img.at<float>(i - 1, j + 1) + \
    img.at<float>(i - 1, j) + img.at<float>(i - 1, j - 1) + img.at<float>(i, j - 1) + img.at<float>(i + 1, j - 1))/8;
}

// The following function is a mean interpolation function fitted to interpolate the missing data after 
// the similarity transformation
cv::Mat_<float> u::motionInterpolation(cv::Mat_<float> img, float threshold){
    cv::Mat_<float> tmp(img.size().height, img.size().width);
    img.copyTo(tmp);

    for (int i{1}; i < img.size().height - 1; i++){
        for (int j{1}; j < img.size().width - 1; j++){
            float mean = meanAroundPixel(tmp, i, j);
            // if the pixel is white and if the mean value of the surrounding pixels is below 
            // the threshold then the pixel takes the value of the mean of the surrounding pixels
            if ((tmp.at<float>(i, j) == 1) and (mean < threshold)){
                img.at<float>(i, j) = mean;
            }
            // same reasonning for black pixels
            else if ((tmp.at<float>(i, j) == 0) and (mean > 1 - threshold)){
                img.at<float>(i, j) = mean;
            }
        }
    }

    // We also treat the borders seperatly to avoid artefacts

    // west border
    for (int i{1}; i < img.size().height - 1; i++){
        if ((tmp.at<float>(i, 0) == 1) and (img.at<float>(i - 1, 0) + img.at<float>(i + 1, 0))/2 < threshold){
            img.at<float>(i, 0) = (img.at<float>(i - 1, 0) + img.at<float>(i + 1, 0))/2;
        }
    }
    // east border
    for (int i{1}; i < img.size().height - 1; i++){
        if ((tmp.at<float>(i, img.size().width - 1) == 1) and (img.at<float>(i - 1, img.size().width - 1) + img.at<float>(i + 1, img.size().width - 1))/2 < threshold){
            img.at<float>(i, img.size().width - 1) = (img.at<float>(i - 1, img.size().width - 1) + img.at<float>(i + 1, img.size().width - 1))/2;
        }
    }
    // north border
    for (int i{1}; i < img.size().width - 1; i++){
        if ((tmp.at<float>(0, i) == 1) and (img.at<float>(0, i - 1) + img.at<float>(0, i + 1))/2 < threshold){
            img.at<float>(0, i) = (img.at<float>(0, i - 1) + img.at<float>(0, i + 1))/2;
        }
    }
    // south border
    for (int i{1}; i < img.size().width - 1; i++){
        if ((tmp.at<float>(img.size().height - 1, i) == 1) and (img.at<float>(img.size().height - 1, i - 1) + img.at<float>(img.size().height - 1, i + 1))/2 < threshold){
            img.at<float>(img.size().height - 1, i) = (img.at<float>(img.size().height - 1, i - 1) + img.at<float>(img.size().height - 1, i + 1))/2;
        }
    }

    return img;
}

/*!
 *  A function which performs spatial convolution of two matrices. The result matrix has the same size as the first parameter
 *  of the function, the matrix A, which should be the image matrix.
 *  The second parameter of the function should be the filter matrix F.
 * \param[in] const cv::Mat & Image matrix, const cv::Mat & Filter matrix
 * \param[out] cv::Mat Convolution product
*/
cv::Mat u::convolveSpatial(const cv::Mat &A, const cv::Mat &F)
{
	if (A.cols > F.cols && A.rows > F.rows && F.rows == F.cols && F.rows % 2 != 0){
        	auto start = std::chrono::system_clock::now();

       		// creating padded A matrix
		cv::Mat paddedA = cv::Mat::ones(A.rows + 2*(F.rows - 1), A.cols + 2*(F.cols - 1),CV_32F);
		for (size_t i{0}; i < A.rows; i++){
			for(size_t j{0}; j < A.cols; j++){
				paddedA.at<float>(i + F.rows - 1,j + F.cols - 1) = A.at<float>(i,j);
			}
		}
		
		// creating rotated F matrix
		cv::Mat rotatedF = cv::Mat::zeros(F.rows,F.cols,CV_32F);
		for (size_t i{0}; i < F.rows; i++){
			for(size_t j{0}; j < F.cols; j++){
				rotatedF.at<float>(i,j) = F.at<float>(F.rows - 1 - i, F.cols - 1 - j);
			}
		}
		
		// creating padded result matrix
		cv::Mat paddedResultM = cv::Mat::ones(paddedA.rows,paddedA.cols,CV_32F);
		int offset{(F.rows - 1)/2};
		for (int i{offset}; i < paddedA.rows - offset; i++){
			for(int j{offset}; j < paddedA.cols - offset; j++){
				float res{0};
				for(size_t k{0}; k < F.rows; k++){
					for (size_t l{0}; l < F.rows; l++){
						res += paddedA.at<float>(i - offset + k,j - offset + l)*rotatedF.at<float>(k,l);
					}
				}
				paddedResultM.at<float>(i,j) = res;
			}
		}
		
		// unpadding result matrix
		cv::Mat resultM = cv::Mat::ones(A.rows,A.cols,CV_32F);
		for (size_t i{0}; i < A.rows; i++){
			for(size_t j{0}; j < A.cols; j++){
				resultM.at<float>(i,j) = paddedResultM.at<float>(i + F.rows - 1, j + F.cols - 1);
			}
		}
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<float> elapsed_seconds = end-start;
		std::time_t end_time = std::chrono::system_clock::to_time_t(end);

		std::cout << "Spatial convolution finished computation at " << std::ctime(&end_time)
		          << "elapsed time: " << elapsed_seconds.count() << "s\n";
		//cv::imshow("Blurred with spatial convolution.", resultM);
		//cv::waitKey(0);
		return resultM;	
	} else{
		throw std::runtime_error("Invalid input. The filter should be a square matrix with an odd number of rows and columms. Furthermore, it should also be smaller than the image matrix.");
	}
}

/*!
 *  A function which performs frequential convolution of two matrices with zero-padding
 *  The result matrix has the same size as the first parameter
 * \param[in] const cv::Mat & Image matrix, const cv::Mat & Filter matrix, bool deconvolve
 * \param[out] cv::Mat Convolution product
*/
cv::Mat u::convolveFrequential(const cv::Mat &image, const cv::Mat &filter, bool deconvolve) {
    /*https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gadd6cf9baf2b8b704a11b5f04aaf4f39d*/
    /*If once it stops working above is stable version*/
    auto start = std::chrono::system_clock::now();
    std::string window_name("");

    cv::Mat convolutionProduct;
    convolutionProduct.create(abs(image.rows), abs(image.cols), image.type());

    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(image.cols + filter.cols - 1);
    dftSize.height = cv::getOptimalDFTSize(image.rows + filter.rows - 1);

    cv::Mat ftImage = cv::Mat::zeros(dftSize.height, dftSize.width,  image.type());
    cv::Mat ftFilter = cv::Mat::zeros(dftSize.height, dftSize.width,  image.type());

    cv::Mat roiA(ftImage, cv::Rect(0,0, image.cols, image.rows));
    image.copyTo(roiA);
    cv::Mat roiB(ftFilter, cv::Rect(0, 0, filter.cols, filter.rows));
    filter.copyTo(roiB);

    cv::dft(ftImage, ftImage, cv::DFT_REAL_OUTPUT);
    cv::dft(ftFilter, ftFilter, cv::DFT_REAL_OUTPUT);

    if(deconvolve){
        window_name = "Deblurred";
        cv::Mat div = cv::Mat::zeros(dftSize.height, dftSize.width,  image.type());
        u::divSpectrums(ftImage, ftFilter, div, 0, false);
        ftImage = div;
    }else{
        window_name = "Blurred";
        cv::mulSpectrums(ftImage, ftFilter, ftImage, 0);
    }

    cv::dft(ftImage, ftImage, cv::DFT_INVERSE + cv::DFT_SCALE);
    ftImage(cv::Rect(0, 0, convolutionProduct.cols, convolutionProduct.rows)).copyTo(convolutionProduct);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "DFT convolution finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    cv::imshow(window_name, convolutionProduct);
    cv::waitKey(0);

    if(deconvolve){
        //cv::normalize(convolutionProduct, convolutionProduct, 0, 1, cv::NORM_MINMAX);
    }
    std::cout << "size = " << convolutionProduct.size().width << " x " << convolutionProduct.size().height << std::endl;
    return convolutionProduct;
}

/*!
 *  A function which performs division of two CCS-packed matricex.
 * \param[in] cv::InputArray first matrix, cv::InputArray second matrix, cv::OutputArray product of division, int flags, bool conjB
*/
static void u::divSpectrums( cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB)
{
    cv::Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;
    std::cout << "CN = " << cn << std::endl;

    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    _dst.create( srcA.rows, srcA.cols, type );
    cv::Mat dst = _dst.getMat();

    CV_Assert(dst.data != srcA.data); // non-inplace check
    CV_Assert(dst.data != srcB.data); // non-inplace check

    bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
                                                          srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if( is_1d && !(flags & cv::DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    const float* dataA = srcA.ptr<float>();
    const float* dataB = srcB.ptr<float>();
    float* dataC = dst.ptr<float>();
    //!!! Value below is insane important for stability and needs to be automate.
    float eps = 0.075;//FLT_EPSILON; // prevent div0 problems ( FLT_EPSILON = 1.19209e-07)
    size_t stepA = srcA.step/sizeof(dataA[0]);
    size_t stepB = srcB.step/sizeof(dataB[0]);
    size_t stepC = dst.step/sizeof(dataC[0]);
    if( !is_1d && cn == 1 ){ //2d + CCS
        for( k = 0; k < (cols % 2 ? 1 : 2); k++ ){
            if( k == 1 )
                dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
            dataC[0] = dataA[0] / (dataB[0] + eps);
            if( rows % 2 == 0 )
                dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
            for( j = 1; j <= rows - 2; j += 2 ){
                float denom = (float)dataB[j*stepB]*dataB[j*stepB] +
                              (float)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (float)eps;
                float re = (float)dataA[j*stepA]*dataB[j*stepB] +
                           (float)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                float im = (float)dataA[(j+1)*stepA]*dataB[j*stepB] -
                           (float)dataA[j*stepA]*dataB[(j+1)*stepB];

                dataC[j*stepC] = (float)(re / denom);
                dataC[(j+1)*stepC] = (float)(im / denom);
            }
            if( k == 1 )
                dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
        }
    }

    for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC ){
        if( is_1d && cn == 1 ){
            dataC[0] = dataA[0] / (dataB[0] + eps);
            if( cols % 2 == 0 )
                dataC[j1] = dataA[j1] / (dataB[j1] + eps);
        }
        for( j = j0; j < j1; j += 2 ){
            float denom = (float)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
            float re = (float)(dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1]);
            float im = (float)(dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1]);
            dataC[j] = (float)(re / denom);
            dataC[j+1] = (float)(im / denom);
        }
    }
}

/*!
 * A function which sets matrix of spatially invariant filter.
 * \param[in] FilterType filter type, int filter size, float sigma(for Gaussian filter)
 * \param[out] cv::Mat Filter
*/
cv::Mat u::setFilter(FilterType filterType, int size, float sigma) {
    if(size % 2 == 0){
        throw std::runtime_error("Filter size should be odd.");
    }
    if(filterType == FilterType::Sharpening_Sobol_X || filterType == Sharpening_Sobol_Y){size = 3;}

    cv::Mat filter(size, size, CV_32F, cv::Scalar(0.));
    float size2 = float(size*size);

    switch (filterType){
        case FilterType::Average_blurring: {
            filter = cv::Mat::ones(size,size,CV_32F) / size2;
            break;
        }
        case FilterType::Test: {
            int sum = 0;
            for (int x = 0; x < size; ++x){
                for (int y = 0; y < size; ++y) {
                    filter.at<float>(x, y) = sum;
                    sum++;
                }
            }
            break;
        }
        case FilterType::Gaussian_blurring: {
            float mean = size/2;
            float sum{0.0f};
            for (int x = 0; x < size; ++x){
                for (int y = 0; y < size; ++y) {
                    filter.at<float>(x, y) = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0)))
                                             / (2 * M_PI * sigma * sigma);
                    sum += filter.at<float>(x, y);
                }
            }
            for (int x = 0; x < size; ++x)
                for (int y = 0; y < size; ++y)
                    filter.at<float>(x,y) /= sum;
            break;
        }
        case FilterType::Sharpening_Sobol_X: {
            filter.at<float>(0,0) = 1; filter.at<float>(0,1) = 0; filter.at<float>(0,2) = -1;

            filter.at<float>(1,0) = 2; filter.at<float>(1,1) = 0; filter.at<float>(1,2) = -2;

  	        filter.at<float>(2,0) = 1; filter.at<float>(2,1) = 0; filter.at<float>(2,2) = -1;
            break;
        }
        case FilterType::Sharpening_Sobol_Y: {
            filter.at<float>(0,0) = 1; filter.at<float>(0,1) = 2; filter.at<float>(0,2) = 1;

            filter.at<float>(1,0) = 0; filter.at<float>(1,1) = 0; filter.at<float>(1,2) = 0;

            filter.at<float>(2,0) = -1; filter.at<float>(2,1) = -2; filter.at<float>(2,2) = -1;
            break;
        }
        default : {
            throw std::runtime_error("Such filter does not exist.");
        }
    }
    return filter;
}

/*!
 * A function which sets matrix of spatially variant filter.
 * \param[in] FilterType filter type, int filter size, float sigma(for Gaussian filter), cv::Point barycente of image, Window window which is corresponding to that filter
 * \param[out] cv::Mat Filter
*/
cv::Mat u::setSVFilter(FilterType filterType, int size, float sigma, cv::Point barycente, Window window) {
    if(size % 2 == 0){
        throw std::runtime_error("Filter size should be odd.");
    }

    cv::Mat filter(size, size, CV_32F, cv::Scalar(0.));
    float size2 = float(size*size);

    switch (filterType){
        case FilterType::SV_Gaussian_blurring: {
            std::cout << "Barycentre: " << barycente.x << " x " << barycente.y << std::endl;
            std::cout << "WIndow centre: " << window.centre.x << " x " << window.centre.y << std::endl;
            float mean = size/2;
            float sum{0.0f};
            for (int x = 0; x < size; ++x){
                for (int y = 0; y < size; ++y) {
                    filter.at<float>(x, y) = exp(-0.5 * (pow((window.centre.x - barycente.x) / sigma, 2.0) + pow((window.centre.y - barycente.y) / sigma, 2.0)))
                                             / (2 * M_PI * sigma * sigma);
                    sum += filter.at<float>(x, y);
                    std::cout << "Filter = " << filter.at<float>(x, y) << std::endl;
                }
            }
            for (int x = 0; x < size; ++x)
                for (int y = 0; y < size; ++y)
                    filter.at<float>(x,y) /= sum;
            break;
        }
        case FilterType::SV_Custom_blurring: {
            std::cout << "Barycentre: " << barycente.x << " x " << barycente.y << std::endl;
            std::cout << "WIndow centre: " << window.centre.x << " x " << window.centre.y << std::endl;
            float mean = size/2;
            float sum{0.0f};
            for (int x = 0; x < size; ++x){
                for (int y = 0; y < size; ++y) {
                    filter.at<float>(x, y) = (pow((window.centre.x + x - barycente.x)/sigma, 2.0) + pow((window.centre.y + y - barycente.y)/sigma, 2.0))/ (2 * M_PI * sigma * sigma);
                    sum += filter.at<float>(x, y);
                    std::cout << "Filter = " << filter.at<float>(x, y) << std::endl;
                }
            }
            for (int x = 0; x < size; ++x)
                for (int y = 0; y < size; ++y)
                    filter.at<float>(x,y) /= sum;
            break;
        }
        case FilterType::SV_Inverse_Custom_blurring: {
            std::cout << "Barycentre: " << barycente.x << " x " << barycente.y << std::endl;
            std::cout << "WIndow centre: " << window.centre.x << " x " << window.centre.y << std::endl;
            float mean = size/2;
            float sum{0.0f};
            for (int x = 0; x < size; ++x){
                for (int y = 0; y < size; ++y) {
                    filter.at<float>(x, y) =(2 * M_PI * sigma * sigma) / (pow((window.centre.x + x - barycente.x)/sigma, 2.0) + pow((window.centre.y + y - barycente.y)/sigma, 2.0));
                    sum += filter.at<float>(x, y);
                    std::cout << "Filter = " << filter.at<float>(x, y) << std::endl;
                }
            }
            for (int x = 0; x < size; ++x)
                for (int y = 0; y < size; ++y)
                    filter.at<float>(x,y) /= sum;
            break;
        }
        default : {
            throw std::runtime_error("Such filter does not exist.");
        }
    }
    return filter;
}

/*!
 * A function which calculates sigma for Gaussian filter corresponding to the distance to barycentre.
 * \param[in] Window window, const cv::Point& barycentre, int image width, int image height, bool inverse
 * \param[out] float sigma
*/
float u::getSigma(Window window, const cv::Point& barycentre, int imgSizeX, int imgSizeY, bool inverse){
    float sigma{0.0f};
    float coeff{7.0f};
    int rx = abs(window.centre.x - barycentre.x);
    int ry = abs(window.centre.y - barycentre.y);
    int maxdistX = std::max(abs(0 - barycentre.x), abs(imgSizeX - barycentre.x));
    int maxdistY = std::max(abs(0 - barycentre.y), abs(imgSizeY - barycentre.y));
    std::cout << "Max dist: " << maxdistX << " x " << maxdistY << std::endl;
    std::cout << "Dist: " << rx << " x " << ry << std::endl;

    int maxRad = maxdistX*maxdistX + maxdistY*maxdistY;
    int rad = rx*rx + ry*ry;
    std::cout << "Rad = " << rad << std::endl;
    std::cout << "MaxRad= " << maxRad << std::endl;
    std::cout << "Rad/MaxRad= " <<float(rad)/float(maxRad)<< std::endl;

    if(inverse){
        sigma = 2.0f - coeff*float(rad)/float(maxRad);

    }else{
        sigma = coeff*float(rad)/float(maxRad) + 1.0f;
    }
    std::cout << "Window centre = " << window.centre.x << " x " << window.centre.y << std::endl;
    std::cout << "Sigma = " << sigma << std::endl;
    return sigma;
}

/*!
 * A function which fills vector of widows with corresponding windows.
 * \param[in] std::vector<Window>& windows, const std::vector<cv::Point>& centres, const cv::Mat& image
*/
void u::setWindows(std::vector<Window>& windows, const std::vector<cv::Point>& centres, const cv::Mat& image) {
    windows.reserve(centres.size());
    cv::Mat showWindows = image;
    for (const auto & centre : centres) {
        Window window;
        window.centre = centre;
        window.window = u::getWindow(centre, window.sizeX, window.sizeY, image, showWindows);
        windows.push_back(window);
    }
}

/*!
 * A function which takes window from image.
 * \param[in] const cv::Point& position, int sizeX, int sizeY, const cv::Mat& image, cv::Mat& showWindows
*/
cv::Mat u::getWindow(const cv::Point& position, int sizeX, int sizeY, const cv::Mat& image, cv::Mat& showWindows) {
    cv::Mat window(sizeY, sizeX, CV_32F, cv::Scalar(0.));
    for (int i = position.y - sizeY/2; i < position.y + sizeY/2; ++i) {
        for (int j = position.x - sizeX/2; j < position.x + sizeX/2; ++j) {
            window.at<float>(i - position.y + sizeY/2,j - position.x + sizeX/2) = image.at<float>(i,j);
        }
    }
    return window;
}

/*!
 * A function which emplace windows back in image.
 * If two windows are intercrossing it just rewrites previous window with a new one
 * \param[in] std::vector<Window> &windows, cv::Mat &image
*/
void u::emplaceWindows(std::vector<Window> &windows, cv::Mat &image) {
    int counter = 0;
    for (auto & window : windows) {
        counter++;
        //to highlight borders of a window
        for (int i = 0; i < window.sizeY; ++i) {
            for (int j = 0; j < window.sizeX; ++j) {
                if(i == 0 || i == window.sizeY -1 || j == 0 || j == window.sizeX - 1 || i == 1 || i == window.sizeY - 2 || j == 1 || j == window.sizeX - 2){
                    window.window.at<float>(i,j) = 0.5;
                }
            }
        }//end of highlighting
        for (int i = window.centre.y - window.sizeY/2; i < window.centre.y + window.sizeY/2; ++i) {
            for (int j = window.centre.x - window.sizeX/2; j < window.centre.x + window.sizeX/2; ++j) {
                image.at<float>(i,j) =  window.window.at<float>(i - window.centre.y + window.sizeY/2,j - window.centre.x + window.sizeX/2);
            }
        }
    }
}

/*!
 * A function which emplace windows back in image.
 * If two windows are intercrossing it calculates weighted average of intercrossing windows.
 * \param[in] std::vector<Window> &windows, cv::Mat &image
*/
void u::smartEmplaceWindows(std::vector<Window>& windows, cv::Mat& image){
    cv::Point p;
    std::vector<int> indexes;
    std::vector<float> r;
    float windowR, sumWeights{0.0f};
    std::vector<float> weights;
    int index;
    for (int i = 0; i < image.size().height; ++i) {
        for (int j = 0; j < image.size().width; ++j) {
            p.x = j; p.y = i;
            indexes = u::indexWindowPoint(p, windows);
            if(indexes.empty()){continue;}
            else{r  = u::getRads(p, windows, indexes); image.at<float>(i,j) = 0.;}

            for (int k = 0; k < indexes.size(); ++k) {
                index = indexes[k];
                windowR = sqrtf(float(windows[index].sizeX*windows[index].sizeX)/4.0f + float(windows[index].sizeY*windows[index].sizeY)/4.0f);
                weights.push_back(-(r[k]/windowR) + 1);
                sumWeights += -(r[k]/windowR) + 1;
            }//foreach corresponding window

            for (float & weight : weights) {
                weight /= sumWeights;
            }

            for (int k = 0; k < indexes.size(); ++k) {
                image.at<float>(i,j) += u::getValueOfWindow(p, windows[indexes[k]])*weights[k];
            }
            indexes.resize(0);
            r.resize(0);
            weights.resize(0);
            sumWeights = 0.0f;
        }//foreach pixel
    }
};


/*!
 * A function which checks if two windows are intercrossing.
 * \param[in] const Window &window1, const Window &window2
 * \param[out] bool intercrossing or not
*/
bool u::isWindowsIntercross(const Window &window1, const Window &window2) {
    if(window1.window.data == window2.window.data){
        return false;
    }
    int disXCentres = abs(window1.centre.x - window2.centre.x);
    int disYCentres = abs(window1.centre.y - window2.centre.y);
    int sideX = (window1.sizeX + window2.sizeX)/2;
    int sideY = (window1.sizeY + window2.sizeY)/2;
    if((disXCentres < sideX) && (disYCentres < sideY)){
        return true;
    }
    else{
        return false;
    }
}

/*!
 * A function which checks if the point belongs to the window.
 * \param[in] const cv::Point &point, const Window &window
 * \param[out] bool belongs or not
*/
bool u::isPointInWindow(const cv::Point &point, const Window &window) {
    int disXCentres = abs(window.centre.x - point.x);
    int disYCentres = abs(window.centre.y - point.y);
    int sideX = (window.sizeX)/2;
    int sideY = (window.sizeY)/2;
    if((disXCentres < sideX) && (disYCentres < sideY)){
        return true;
    }
    else{
        return false;
    }
}

/*!
 * A function which creates array of indexes. Each index corresponds to window of array of windows to which the point belongs to.
 * \param[in] cv::Point &point,const std::vector<Window> &windows
 * \param[out] std::vector<int> Indexes of windows containing that point
*/
std::vector<int> u::indexWindowPoint(cv::Point &point,const std::vector<Window> &windows) {
    std::vector<int> indexes;
    int counter{0};
    for (auto & window : windows) {
        if(isPointInWindow(point, window)){
            indexes.push_back(counter);
        }
        ++counter;
    }
    return indexes;
}

/*!
 * A function which creates array of radiuses. Each radius corresponds to distance between point and centre of indexed window.
 * \param[in] const cv::Point& point, const std::vector<Window>& windows, const std::vector<int>& indexes
 * \param[out] std::vector<float> Radiuses to the centres of windows containing that point
*/
std::vector<float> u::getRads(const cv::Point& point, const std::vector<Window>& windows, const std::vector<int>& indexes){
    std::vector<float> rads;
    int index{0}, a{0}, b{0};
    float r{0};
    for (int i = 0; i < indexes.size(); ++i) {
        index = indexes[i];
        a = point.x - windows[i].centre.x;
        b = point.y - windows[i].centre.y;
        r = sqrtf(float(a*a) + float(b*b));
        rads.push_back(r);
    }
    return rads;
}


/*!
 * A function which returns value of a window. Point coordinates are in image's sistem and this function recalculates coordinates in window's basis.
 * \param[in] const cv::Point& point, Window& window
 * \param[out] float Value
*/
float u::getValueOfWindow(const cv::Point& point, Window& window){
    int x = point.x - window.centre.x + window.sizeX/2 ;
    int y = point.y - window.centre.y + window.sizeY/2;
    return window.window.at<float>(y,x);
}

/*!
 *  A function which performs frequential convolution of two matrices without zero-padding
 *  The result matrix has the same size as the first parameter
 * \param[in] const cv::Mat & Image matrix, const cv::Mat & Filter matrix, bool deconvolve
 * \param[out] cv::Mat Convolution product
*/
cv::Mat u::expConvolveFrequential(const cv::Mat &image, const cv::Mat &filter, bool deconvolve) {
    auto start = std::chrono::system_clock::now();
    std::string window_name("");

    cv::Mat convolutionProduct;
    convolutionProduct.create(image.rows, image.cols, image.type());

    cv::Mat ftImage; image.copyTo(ftImage);
    cv::Mat ftFilter = cv::Mat::zeros(image.rows, image.cols,  image.type());
    u::splitFilter(filter, ftFilter);

    cv::Mat planesImage[] = {ftImage, cv::Mat::zeros(ftImage.size(), CV_32F)};
    cv::Mat complexIimage;
    merge(planesImage, 2, complexIimage);

    cv::Mat planesFilter[] = {ftFilter, cv::Mat::zeros(ftFilter.size(), CV_32F)};
    cv::Mat complexIFilter;
    merge(planesFilter, 2, complexIFilter);

    cv::dft(complexIimage, complexIimage,cv::DFT_COMPLEX_INPUT +  cv::DFT_COMPLEX_OUTPUT);
    cv::dft(complexIFilter, complexIFilter, cv::DFT_COMPLEX_INPUT +  cv::DFT_COMPLEX_OUTPUT);

    cv::Mat resReal, resComplex;
    cv::Mat complexIRes;

    if(deconvolve){
        window_name = "Deblurred";
        split(complexIimage, planesImage);
        split(complexIFilter, planesFilter);
        u::div(planesImage[0], planesImage[1],planesFilter[0], planesFilter[1], resReal, resComplex);

        cv::Mat planesRes[] = {resReal, resComplex};
        merge(planesRes, 2, complexIRes);
    }else{
        window_name = "Blurred";
        split(complexIimage, planesImage);
        split(complexIFilter, planesFilter);
        u::mul(planesImage[0], planesImage[1],planesFilter[0], planesFilter[1], resReal, resComplex);

        cv::Mat planesRes[] = {resReal, resComplex};
        merge(planesRes, 2, complexIRes);
    }

    cv::dft(complexIRes, ftImage, cv::DFT_INVERSE + cv::DFT_SCALE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_COMPLEX_INPUT);
    split(ftImage, planesImage);
    planesImage[0](cv::Rect(0, 0, convolutionProduct.cols, convolutionProduct.rows)).copyTo(convolutionProduct);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "DFT convolution finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    cv::imshow(window_name, convolutionProduct);
    cv::waitKey(0);

    return convolutionProduct;
}


cv::Mat_<float> u::horizontalDerivative(cv::Mat_<float> img){
    cv::Mat_<float> tmp(img.size().height, img.size().width);
    for (int i = 0; i < img.size().height; i++){
        for (int j = 0; j < img.size().width; j++){
            if (j == 0){
                tmp.at<float>(i, j) = img.at<float>(i, j + 1) - img.at<float>(i, j);
            }
            else if (j == img.size().width - 1)
            {
                tmp.at<float>(i, j) = img.at<float>(i, j) - img.at<float>(i, j - 1);
            }
            else{
            tmp.at<float>(i, j) = (img.at<float>(i, j + 1) - img.at<float>(i, j - 1))/2;
            }
        }
    }
    return tmp/(img.size().width*img.size().height);
}

cv::Mat_<float> u::verticalDerivative(cv::Mat_<float> img){
    cv::Mat_<float> tmp(img.size().height, img.size().width);
    for (int i = 0; i < img.size().height; i++){
        for (int j = 0; j < img.size().width; j++){
            if (i == 0){
                tmp.at<float>(i, j) = img.at<float>(i + 1, j) - img.at<float>(i, j);
            }
            else if (i == img.size().height - 1)
            {
                tmp.at<float>(i, j) = img.at<float>(i, j) - img.at<float>(i - 1, j);
            }
            else{
            tmp.at<float>(i, j) = (img.at<float>(i + 1, j) - img.at<float>(i - 1, j))/2;
            }
        }
    }
    return tmp/(img.size().width*img.size().height);
}

double u::nablaPx(cv::Mat_<float> img, cv::Mat_<float> other){
    cv::Mat_<float> der_g = u::horizontalDerivative(img);
    double res = 0;
    for (int i = 0; i < img.size().height; i++){
        for (int j = 0; j < img.size().width; j++){
            res += 2*der_g.at<float>(i, j)*(other.at<float>(i, j) - img.at<float>(i, j));
        }
    }
    return res;
}

double u::nablaPy(cv::Mat_<float> img, cv::Mat_<float> other){
    cv::Mat_<float> der_g = u::verticalDerivative(img);
    double res = 0;
    for (int i = 0; i < img.size().height; i++){
        for (int j = 0; j < img.size().width; j++){
            res += 2*der_g.at<float>(i, j)*(other.at<float>(i, j) - img.at<float>(i, j));
        }
    }
    return res;
}

<<<<<<< HEAD
cv::Point matchBarycentres(cv::Mat_<float> img, cv::Mat_<float> other){
    cv::Point bar_1 = barycentre(img);
    cv::Point bar_2 = barycentre(other);
=======
cv::Point u::matchBarycentres(cv::Mat_<float> img, cv::Mat_<float> other){
    cv::Point bar_1 = u::barycentre(img);
    std::cout << bar_1 << std::endl;
    cv::Point bar_2 = u::barycentre(other);
    std::cout << bar_2 << std::endl;
>>>>>>> fb247fcde6245e203ec3db532d4e7aa8ee5d1339
    // we have to inverst the sign of the y translation for it to make sense
    cv::Point p(int(bar_1.x - bar_2.x), int(bar_2.y - bar_1.y));
    return p;
}

/*!
 *  A function which performs multiplication of two complex matrices
 *  Each matrix is presented by it's real and complex matrices (M1 = M1Real + i*M1Complex)
 *  The result matrix has the same size as the first parameter .
 * \param[in] const cv::Mat &M1Real, const cv::Mat &M1Complex, const cv::Mat &M2Real, const cv::Mat &M2Complex,
 * cv::Mat& ResReal, cv::Mat& ResComplex
*/
void u::mul(const cv::Mat &M1Real, const cv::Mat &M1Complex, const cv::Mat &M2Real, const cv::Mat &M2Complex, cv::Mat& ResReal, cv::Mat& ResComplex){
    int width = M1Real.size().width;
    int height = M1Real.size().height;

    ResReal = cv::Mat::zeros(width, height, CV_32F);
    ResComplex = cv::Mat::zeros(width, height, CV_32F);
    float a1,a2,b1,b2;
    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){
            a1 = M1Real.at<float>(i,j);
            a2 = M2Real.at<float>(i,j);
            b1 = M1Complex.at<float>(i,j);
            b2 = M2Complex.at<float>(i,j);
            ResReal.at<float>(i,j) = a1*a2 - b1*b2;
            ResComplex.at<float>(i,j) = a1*b2 + b1*a2;
        }
    }
}

/*!
 *  A function which performs division of two complex matrices
 *  Each matrix is presented by it's real and complex matrices (M1 = M1Real + i*M1Complex)
 *  The result matrix has the same size as the first parameter .
 * \param[in] const cv::Mat &M1Real, const cv::Mat &M1Complex, const cv::Mat &M2Real, const cv::Mat &M2Complex,
 * cv::Mat& ResReal, cv::Mat& ResComplex
*/
void u::div(const cv::Mat &M1Real, const cv::Mat &M1Complex, const cv::Mat &M2Real, const cv::Mat &M2Complex, cv::Mat& ResReal, cv::Mat& ResComplex){
    int width = M1Real.size().width;
    int height = M1Real.size().height;

    ResReal = cv::Mat::zeros(width, height, CV_32F);
    ResComplex = cv::Mat::zeros(width, height, CV_32F);

    float x1,x2, y1, y2, denom, eps{0.025f};
    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){
            x1 = M1Real.at<float>(i,j);
            y1 = M1Complex.at<float>(i,j);
            x2 = M2Real.at<float>(i,j);
            y2 = M2Complex.at<float>(i,j);
            denom = x2*x2 + y2*y2 + eps;
            ResReal.at<float>(i,j) = (x1*x2 + y1*y2) / denom;
            ResComplex.at<float>(i,j) =(x2*y1 - x1*y2) / denom;
        }
    }
}

/*!
 *  A function which fills filter in a matrix of image's size to perform convolution.
 *  The idea of splitting is taken here: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.472.2396&rep=rep1&type=pdf
 *  The result filter has the same size as the second parameter.
 *  !!! Some test outputs still needed because we're not sure if it works perfectly.
 * \param[in] const cv::Mat& filter, cv::Mat& resFilter
*/
void u::splitFilter(const cv::Mat& filter, cv::Mat& resFilter){
    int halfSize = filter.size().height/2 + 1;
    int size = filter.size().height;
    /*std::cout << "Half size = " << halfSize << std::endl;
    std::cout << "Size = " << size << std::endl;
    std::cout << "Res filter Size = " << resFilter.size().width  << " x " << resFilter.size().height << std::endl;
    std::cout << "Start of Filter" << std::endl;
    std::cout << filter << std::endl;
    std::cout << "End of Filter" << std::endl;*/

    /*first quadrant*/
   // std::cout << "FIrst q" << std::endl;
    for (int i = 0; i < halfSize; ++i) {
        for (int j = 0; j < halfSize; ++j) {
            resFilter.at<float>(i,j) = filter.at<float>(i + halfSize - 1,j + halfSize - 1);
            //std::cout << "i = " << i  << " j =  " << j << " = " << filter.at<float>(i + halfSize,j + halfSize) << std::endl;
        }
    }
    /*second quadrant*/
    //std::cout << "Second q" << std::endl;
    for (int i = 0; i < halfSize; ++i) {
        for (int j = halfSize; j < size; ++j) {
            resFilter.at<float>(i,resFilter.size().width - size + j) = filter.at<float>(i + halfSize - 1,j - halfSize);
            //std::cout << "i = " << i << " j =  " << resFilter.size().width - size + j << " = " << filter.at<float>(i + halfSize,j - halfSize) <<std::endl;
        }
    }
    /*third quadrant*/
    //std::cout << "Third q" << std::endl;
     for (int i = halfSize; i < size; ++i) {
         for (int j = 0; j < halfSize; ++j) {
             resFilter.at<float>(i + resFilter.size().height - size,j) = filter.at<float>(i - halfSize,j + halfSize -1);
             //std::cout << "i = " << i + resFilter.size().height - size << " j =  " << j << " = " << filter.at<float>(i - halfSize,j + halfSize) <<std::endl;
         }
     }
     /*fourth quadrant*/
    //std::cout << "fourth q" << std::endl;
    for (int i = halfSize; i < size; ++i) {
        for (int j = halfSize; j < size; ++j) {
            resFilter.at<float>(i + resFilter.size().height - size,j + resFilter.size().width - size) = filter.at<float>(i - halfSize ,j - halfSize);
            //std::cout << "i = " << i + resFilter.size().height - size << " j =  " << j + resFilter.size().width - size << " = " << filter.at<float>(i - halfSize,j - halfSize) << std::endl;
        }
    }
    /*std::cout << "Start of resFilter" << std::endl;
    std::cout << resFilter << std::endl;
    std::cout << "End of resFilter" << std::endl;*/
    //int a;
    //std::cin >> a;
}
