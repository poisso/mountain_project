#include "Image.h"


Image& Image::similarity(double theta, double scale, double tx, double ty){

    // Now we calcute the similarity matrix to perform the transform
    cv::Mat_<float> sim_mat(2, 3);
    sim_mat(0, 0) = cos(theta)*scale;
    sim_mat(0, 1) = -sin(theta)*scale;
    sim_mat(1, 0) = sin(theta)*scale;
    sim_mat(1, 1) = cos(theta)*scale;
    sim_mat(0, 2) = ty;
    sim_mat(1, 2) = tx;


    cv::Mat_<float> tmp(height, width);
    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){
            tmp.at<float>(i, j) = 1;
        }
    }

    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){

            cv::Mat_<float> res =  u::pixelRotation(i - round(height/2), j - round(width/2), sim_mat);

            // to reduce floating point mistake we round the coordinates to the closest integer (better than just using floor function)
            //cv::Point_<int> new_coordinates(round(res.at<float>(0, 0)), round(res.at<float>(1, 0)));
            int new_x = res(0, 0) + round(height/2);
            int new_y = res(1, 0) + round(width/2);
            if ((new_x >= 0) and (new_x < height) and (new_y >= 0) and (new_y < width)){
                tmp.at<float>(new_x, new_y) = matrixImg.at<float>(i, j);
            }
        }
    }

    matrixImg = u::motionInterpolation(tmp, 0.99);
    //save("../similarity_transform.png");
    return *this;
}

Image& Image::isotropicLocalSimilarity(double theta, double radius, int x, int y){
    // We create the centre of rotation
    cv::Point centre(x, y);

    // We create the temporary matrix in which we will store the result
    cv::Mat_<float> tmp(height, width);

    // Declaration of the rotation matrix which will be changed for each pixel
    cv::Mat_<float> sim_mat(2, 3);

    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){
            // We change of basis

            int Y = i - centre.x;
            int X = j - centre.y;

            double distance_to_centre_of_rotation_squared = (X*X) + (Y*Y);
            if (distance_to_centre_of_rotation_squared <= radius*radius){
                
                // We must first calcultate the angle of rotation depending on the distance to the centre
                double alpha = theta*exp(-(2*log(10)/pow(radius, 2))*distance_to_centre_of_rotation_squared);
                
                // We test different functions to do so:
                
                // double alpha = theta*(1 - sqrt(distance_to_centre_of_rotation_squared)/radius);
                // double alpha = theta*exp(-(2*log(10)/radius)*sqrt(distance_to_centre_of_rotation_squared));
                // double alpha = theta*exp(-(2*log(10)/pow(radius, 4))*pow(distance_to_centre_of_rotation_squared, 2));
                
                // Now we calcute the similarity matrix to perform the transform

                sim_mat(0, 0) = cos(alpha);
                sim_mat(0, 1) = -sin(alpha);
                sim_mat(1, 0) = sin(alpha);
                sim_mat(1, 1) = cos(alpha);
                sim_mat(0, 2) = 0;
                sim_mat(1, 2) = 0;

                // We rotate the pixel and we get its neww coordinate

                cv::Mat_<float> res =  u::pixelRotation(Y, X, sim_mat);
                
                // We change back to the usual image basis
                cv::Point_<float> new_coordinates(res(0, 0) + centre.x , res(1, 0) + centre.y);
                int new_x = new_coordinates.x;
                int new_y = new_coordinates.y;

                // We make sure the new pixel is in the boundaries of the image
                if ((new_x >= 0) and (new_x < height) and (new_y >= 0) and (new_y < width)){
                    tmp.at<float>(new_x, new_y) = matrixImg.at<float>(i, j);
                }
            }
            else{
                tmp.at<float>(i, j) = matrixImg.at<float>(i, j);
            }
        }
    }
    matrixImg = u::motionInterpolation(tmp, 0.99);
    save("../isotropic_transform.png");
    return *this;
}


Image& Image::anisotropicLocalSimilarity(double theta,  double ellipse_width, double ellipse_height, int x, int y){
    // We create the temporary matrix in which we will store the result
    cv::Mat_<float> tmp(height, width);
    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){
            tmp.at<float>(i, j) = 0;
        }
    }
    cv::Point centre(x, y);
    // Declaration of the rotation matrix which will be changed for each pixel
    cv::Mat_<float> sim_mat(2, 3);

    for (int i{0}; i < height; i++){
        for (int j{0}; j < width; j++){
            // We change coordinates w.r.t. cetre of ellipse (cv::Point centre):
            int Y = i - centre.y;
            int X = j - centre.x;

            //We calculate width and size of ellipse to which given point belongs:
            double ellipse_scale_coef = (X*X)/(ellipse_width*ellipse_width) + (Y*Y)/(ellipse_height*ellipse_height);
            double new_width_carre  = ellipse_width*ellipse_width*ellipse_scale_coef;
            double new_height_carre = ellipse_height*ellipse_height*ellipse_scale_coef;

            //We calculate summary distance of point to de focis:
            double distance_to_focis = 2*sqrt(new_width_carre);

            if (distance_to_focis <= 2*ellipse_width){
                // We must first calcultate the angle of rotation depending on the distance to the focis
                double alpha = theta*exp(-(2*log(10)/pow(2*ellipse_width, 2))*pow(distance_to_focis, 2));

                //There are several bad versions for the report:
                
                //double alpha = theta*(1 - sqrt(distance_to_focis)/2*ellipse_width);
                //double alpha = exp(-distance_to_focis)*(-the ta/(exp(-2*ellipse_width)+1)) + theta - (-theta/(exp(-2*ellipse_width)+1));
                //double alpha = theta*exp(-pow(distance_to_focis, 2)/(2*pow(2*ellipse_width, 2)));
                //double alpha = exp(-distance_to_focis)*(-theta/(exp(-2*ellipse_width)+1)) + theta - (-theta/(exp(-2*ellipse_width)+1));
                //double alpha = theta*exp((-2*log(10))*(pow(Y/new_height_carre, 2) + pow(X/new_width_carre, 2)));
                //double alpha = 15;
                //double alpha = theta*(1 - sqrt(distance_to_focis)/2*width_a)
                
                // Now we calcute the similarity matrix to perform the transform

                sim_mat(0, 0) = cos(alpha);
                sim_mat(0, 1) = -sin(alpha);
                sim_mat(1, 0) = sin(alpha);
                sim_mat(1, 1) = cos(alpha);
                sim_mat(0, 2) = 0;
                sim_mat(1, 2) = 0;

                // Now we calcute the similarity matrix to perform the transform
                double scale = 1;
                double tx = 0;
                double ty = 0;
                sim_mat(0, 0) = cos(alpha)*scale;
                sim_mat(0, 1) = -sin(alpha)*scale;
                sim_mat(1, 0) = sin(alpha)*scale;
                sim_mat(1, 1) = cos(alpha)*scale;
                sim_mat(0, 2) = tx;
                sim_mat(1, 2) = ty;

                //We rotate the pixel on the angle alpha:
                cv::Mat_<float> res =  u::pixelRotation(Y, X, sim_mat);
                X = res(1, 0);
                Y = res(0, 0);

                //And now we searching for coefficient to scale obtained vector such the way that it will belong to the same ellipse with the given point:
                double new_ellipce_eq = X*X/(new_width_carre) + Y*Y/(new_height_carre);
                double coef_sqrt = 1/new_ellipce_eq;

                //We change back to the usual coordinates:
                cv::Point_<float> new_coordinates(res(0, 0)*sqrt(coef_sqrt) + centre.y , res(1, 0)*sqrt(coef_sqrt) + centre.x);
                int new_x = new_coordinates.x;
                int new_y = new_coordinates.y;

                // We make sure the new pixel is in the boundaries of the image
                if ((new_x >= 0) and (new_x < height) and (new_y >= 0) and (new_y < width)){
                    tmp.at<float>(new_x, new_y) = matrixImg.at<float>(i, j);
                }
            }
            else{
                tmp.at<float>(i, j) = matrixImg.at<float>(i, j);
            }
        }
    }
    matrixImg = u::motionInterpolation(tmp, 0.99);
    save("../non_isotropic_transform.png");
    return *this;
}