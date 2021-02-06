#include "Image.h"

int Image::lossFunction(Image &other){
    int loss = 0;
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            loss += pow(matrixImg.at<float>(i, j) - other.matrixImg.at<float>(i, j), 2);
        }
    }
    return loss;

}

Image &Image::plotLoss(Image &other){
    cv::Point b = u::barycentre(other.matrixImg);
    std::vector<int> res;
    for (int i = 0; i < (this->width - b.x); i++){
        int l = this->lossFunction(other.similarity(0, 1, 1, 0));
        res.push_back(round(l/30));
    }
    for (int i = 0; i < this->width; i++){
        for (int j = 0; j < this->height; j++)
            if (j == res[i]){
                this->matrixImg.at<float>(this->height - j - 1, i) = 1;
            }
            else{
                this->matrixImg.at<float>(this->height - j - 1, i) = 0;
            }
    }
    return *this;
}

cv::Point Image::gradientDescent(Image &other, double alpha, double epsilon){
    cv::Point initial_translation = u::matchBarycentres(matrixImg, other.matrixImg);
    int px = initial_translation.x;
    int py = initial_translation.y;
    other.similarity(0, 1, px, py);
    double dpx = u::nablaPx(matrixImg, other.matrixImg);
    double dpy = u::nablaPy(matrixImg, other.matrixImg);
    int i = 0;
    while ((fabs(dpx) > epsilon) and (fabs(dpy) > epsilon)) {
        other.similarity(0, 1, -alpha*dpx, -alpha*dpy);
        if (i%10 == 0){
            other.show();
        }
<<<<<<< HEAD
        dpx = nablaPx(matrixImg, other.matrixImg);
        dpy = nablaPy(matrixImg, other.matrixImg);
        std::cout << "dpx " << dpx << std::endl;
        std::cout << "dpy " << dpy << std::endl;
=======
        dpx = u::nablaPx(matrixImg, other.matrixImg);
        dpy = u::nablaPy(matrixImg, other.matrixImg);
        std::cout << dpx << std::endl;
        std::cout << dpy << std::endl;
>>>>>>> fb247fcde6245e203ec3db532d4e7aa8ee5d1339
        px -= alpha*dpx;
        py -= alpha*dpy;
        i += 1;
    }
    cv::Point p(px, py);
    return p;
}


void Image::uselessPlot(Image &autre){
    cv::Mat_<float> img;
    cv::Mat_<float> other;

    matrixImg.copyTo(img);
    (autre.matrixImg).copyTo(other);

    cv::Mat_<float> tmp(height, width);

    cv::Mat_<float> der_g = u::horizontalDerivative(img);
    for (int i = 0; i < img.size().height; i++){
        for (int j = 0; j < img.size().width; j++){
            tmp.at<float>(i, j) = 2*der_g.at<float>(i, j)*(other.at<float>(i, j) - img.at<float>(i, j));
        }
    }
   

    cv::Mat_<float> der_g_2 = u::verticalDerivative(img);
    double res = 0;
    for (int i = 0; i < img.size().height; i++){
        for (int j = 0; j < img.size().width; j++){
            tmp.at<float>(i, j )= 2*der_g_2.at<float>(i, j)*(other.at<float>(i, j) - img.at<float>(i, j));
        }
    }
    
    tmp.copyTo(matrixImg);
    this->show();
}