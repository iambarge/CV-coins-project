//
//  coins.cpp
//  coins
//
//  Created by Nicolò Bargellesi on 27/06/2019.
//
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <string>
#include "coins_toolbox.cpp"

using namespace std;
using namespace cv;

// custom data structure for circle detection
struct circleParams {
    Mat img_gray;               //  gray scale image for detection
    int Th;                     //  Canny higher threshold (Tl = Th/2)
    int circ_th;                //  circle threshold for HOUGH transform
    int minRadius;              //  minimum radius for circles
    int maxRadius;              //  maximum radius for circles
    vector<Vec3f> circles;      //  resulting circles
};

// load images from the desired folder
//  imgs_directory[100]: images' folder directory
//  imgs_filename[4]:    name scheme shared by images
//  extension[4]:        image type extension
//  num_imgs:            number of images to merge
vector<Mat> loadImages(char imgs_directory[100], char imgs_filename[4], char extension[4], int num_imgs);

// load the coins dataset with the desired feature parameters
//  feat_params:    parameters for feature detection
vector<Coin> loadDataset(FParams feat_params);

// applies gray scale conversion and gaussian blur
//  input_img:      input image
//  kernel_size:    kernel size for the gaussian blur
Mat imgPreproc(Mat input_img, int kernel_size);

// detect edges of a grey scale image through Canny algorithm
//  img_gray:   input image (gray scale)
//  Th:         Canny higher threshold (Tl = Th/2)
Mat detectEdges(Mat img_gray, int Th);

// user interface for circular shapes detection
//  img_gray:   input image (gray scale) for the HOUGH transform
//  Th:         Canny higher threshold (Tl = Th/2)
//  circ_th:    circle threshold for HOUGH transform
vector<Vec3f> detectCircles(Mat img_gray, int Th, int circ_th);

// onChange functions for trackbars
static void onThChange(int Th, void* circ_data);
static void onCircChange(int circ_th, void* circ_data);

// draw circles on the desired image
//  img:        input image
//  circles:    vector of circles to be drawn
void drawCricles(Mat img, vector<Vec3f> circles);

// extract a possible coin from the circle data
//  input_img:  scene with multiple coins
//  circles:    detected circle (possible coin position)
Coin extractCoin(Mat input_img, Vec3f circle);




int main(int argc, char** argv){
    //----- PARAMETERS -----//
    // Pre-Processing parameters
    float kernel_ratio = 1.5;               //  kernel_size = img_size * kernel_ratio[%]
    // Circles detection params
    int Th= 145;                            //  Canny higher threshold (Tl = Th/2)
    int circ_th = 60;                       //  circle threshold for HOUGH transform
    // Feature parameters
    FParams feat_params;
    // ORB parameters
    feat_params.nfeatures = 2500;
    feat_params.scaleFactor = 1.2f;
    feat_params.nlevels = 8;
    feat_params.edgeThreshold = 100;
    feat_params.firstLevel = 0;
    feat_params.WTA_K = 2;
    feat_params.patchSize = 100;
    feat_params.fastThreshold = 10;
    // SIFT parameters
    //nfeatures = 0;
    feat_params.nOctaveLayers = 3;
    feat_params.contrastThreshold = 0.04;
    //edgeThreshold = 10;
    feat_params.sigma = 1.6;
    // SURF parameters
    feat_params.hessianThreshold = 200;
    feat_params.nOctaves = 4;
    //int nOctaveLayers = 3;
    feat_params.extended = true;
    feat_params.upright = false;
    // Matching parameters
    int color_threshold = 45;               //  threshold value for color match
    int max_distance = 100;                 //  max considered match distance (ORB=100; SURF/SIFT=200)
    int RANSAC_th = 1;                      //  maximum allowed reprojection error to treat a point pair as an inlier
    int uncertainty_th = 2;                 //  "NO COIN DETECTED" threshold
    // FEATURE TYPE SELECTION
    feat_params.type = "ORB";               //  ORB / SIFT / SURF
    //----------------------//
    
    // LOAD SCENE
    // Check for input image
    if (argc == 1) {
        cout << "ERROR: Input image need to be passed as argument! \n";
        return 9;
    }
    Mat img = imread(argv[1]);
    Mat res_img = img.clone();
    
    namedWindow("Input Image");
    imshow("Input Image", img);
    cv::waitKey(0);
    
    // LOAD DATASET
    vector<Coin> ref_coins = loadDataset(feat_params);
    
    // PREPROCESSING
    int kernel_size = round(min(img.rows,img.cols)*kernel_ratio/100);
    Mat img_gray = imgPreproc(img, kernel_size);
    
    imshow("Input Image", img_gray);
    cv::waitKey(0);
    
    
    // CIRCLES DETECTION
    vector<Vec3f> circles = detectCircles(img_gray, Th, circ_th);
    drawCricles(res_img, circles);
    
    destroyWindow("Input Image");
    namedWindow("Detected Circles");
    imshow("Detected Circles", res_img);
    cv::waitKey(0);
    destroyWindow("Detected Circles");
    
    // COIN EXTRACTION and CLASSIFICATION
    for (size_t i = 0; i < circles.size(); i++)
    {
        // Extract coin
        Coin detected_coin = extractCoin(img, circles[i]);
        // Guess value
        float guess = detected_coin.guessValue(ref_coins, max_distance, RANSAC_th, uncertainty_th, color_threshold, feat_params);
        
        // Print guess
        char label[5];
        sprintf(label, "%.2f", guess);
        putText(res_img, label, Point(circles[i][0], circles[i][1]), FONT_HERSHEY_COMPLEX_SMALL, circles[i][2]/40, Scalar(255,0,255), 2, LINE_AA);
    }
    
    namedWindow("Results");
    imshow("Results", res_img);
    cv::waitKey(0);
    
    return 0;
}




// load images from the desired folder
vector<Mat> loadImages(char imgs_directory[100], char imgs_filename[4], char extension[4], int num_imgs) {
    vector<Mat> imgs;
    for (int k = 1; k <= num_imgs; k++) {
        char img_file[100];
        sprintf(img_file, "%s/%s%d.%s", imgs_directory, imgs_filename, k, extension);
        Mat img = imread(img_file);
        if (img.dims != 0) {
            imgs.push_back(img);
            //cout << "Loaded image: " << imgs_filename << k << "." << extension << "\n";
        }
        else {
            std::cerr << imgs_filename << k << "." << extension << " NOT FOUND in " << imgs_directory << "\n";
            throw runtime_error("Error in loading images");
        }
    }
    return imgs;
}


// load the coins dataset with the desired feature parameters
vector<Coin> loadDataset(FParams feat_params) {
    char ref_directory[100] = "data/ref";   // default folder location
    char ref_filename[4] = "";              // filename of images (without sufix number)
    char ext[4] = "png";                    // images format extension
    string feat_type = feat_params.type;
    // Load images
    vector<Mat> ref_imgs = loadImages(ref_directory, ref_filename, ext, 8);
    // and corresponding values
    float ref_values[8] = {2,1,0.5,0.2,0.1,0.05,0.02,0.01};
    // Construct coins with desired feature type
    vector<Coin> dataset;
    for(int i=0; i<8; i++){
        Coin ref_coin = Coin(ref_imgs[i], ref_values[i]);
        if(feat_type.compare("ORB")==0) {
            ref_coin.ORBfeatures(feat_params);
        } else if (feat_type.compare("SIFT")==0) {
            ref_coin.SIFTfeatures(feat_params);
        } else if (feat_type.compare("SURF")==0) {
            ref_coin.SURFfeatures(feat_params);
        } else {
            std::cerr << feat_type << " feature type NOT FOUND (it should be ORB / SIFT / SURF). \n";
            throw runtime_error("Error extracting features");
        }
        dataset.push_back(ref_coin);
    }
    return dataset;
}


// applies gray scale conversion and gaussian blur
Mat imgPreproc(Mat input_img, int kernel_size){
    // Gray scale conversion
    Mat img_gray;
    cvtColor(input_img, img_gray, COLOR_BGR2GRAY);
    
    if (kernel_size % 2 == 0) {
        kernel_size++;              // it must be odd
    }
    // Gaussian Filter
    GaussianBlur(img_gray, img_gray, Size(kernel_size, kernel_size), 0, 0 );
    /*
    // Bilateral Filter
    Mat diff, dst;
    Sobel(img_gray, diff, -1, 1, 1);
    bilateralFilter(img_gray, dst, -1, kernel_size, mean(diff)[0]);
    img_gray = dst;
    */
    return img_gray;
}


// detect edges of a grey scale image through Canny algorithm
Mat detectEdges(Mat img_gray, int Th) {
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat detected_edges;
    int Ti = Th/2;
    Canny(img_gray, detected_edges, Ti, Th, 3);
    findContours(detected_edges, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
    // Draw contours
    Mat dst = Mat::zeros(detected_edges.size(), CV_8UC3);
    for( int i = 0; i< contours.size(); i++ )
    {
        drawContours(dst, contours, i ,Scalar( 255 ,255,255 ), 1, 80, hierarchy, INT_MAX, Point(-1 ,-1) );
    }
    return dst;
}


// user interface for circular shapes detection
vector<Vec3f> detectCircles(Mat img_gray, int Th, int circ_th) {
    // Load parameters
    circleParams circ_params;
    circ_params.img_gray = img_gray;
    circ_params.circ_th = circ_th;
    circ_params.Th = Th;
    // Adapt min and max radius to the image size
    circ_params.minRadius = round(min(img_gray.rows,img_gray.cols)/15);
    circ_params.maxRadius = round(min(img_gray.rows,img_gray.cols)/3);
    // Create trackbars
    namedWindow("circles detection");
    createTrackbar("Circle Th", "circles detection", &circ_th, 100, onCircChange, &circ_params);
    createTrackbar("Canny Th", "circles detection", &Th, 200, onThChange, &circ_params);
    onCircChange(circ_th,&circ_params);
    onThChange(Th,&circ_params);
    waitKey(0);
    destroyWindow("circles detection");
    // Return found circles
    return circ_params.circles;
}


// onChange functions for trackbars
static void onThChange(int Th, void* circ_params) {
    Mat img_gray = ((circleParams*)circ_params)->img_gray;
    int circ_th = ((circleParams*)circ_params)->circ_th;
    int minR = ((circleParams*)circ_params)->minRadius;
    int maxR = ((circleParams*)circ_params)->maxRadius;
    
    // Edge Detection
    Mat edgesTest = detectEdges(img_gray, Th);
    
    vector<Vec3f> circles;
    // Circles detection
    HoughCircles(img_gray, circles, HOUGH_GRADIENT, 1, minR, Th, circ_th, minR, maxR);
    drawCricles(edgesTest, circles);
    
    imshow("circles detection", edgesTest);
    ((circleParams*)circ_params)->Th = Th;
    ((circleParams*)circ_params)->circles = circles;
}

static void onCircChange(int circ_th, void* circ_params) {
    Mat img_gray = ((circleParams*)circ_params)->img_gray;
    int Th = ((circleParams*)circ_params)->Th;
    int minR = ((circleParams*)circ_params)->minRadius;
    int maxR = ((circleParams*)circ_params)->maxRadius;
    
    // Edge Detection
    Mat edgesTest = detectEdges(img_gray, Th);
    
    vector<Vec3f> circles;
    // Circles detection
    HoughCircles(img_gray, circles, HOUGH_GRADIENT, 1, minR, Th, circ_th, minR, maxR);
    drawCricles(edgesTest, circles);
    
    imshow("circles detection", edgesTest);
    ((circleParams*)circ_params)->circ_th = circ_th;
    ((circleParams*)circ_params)->circles = circles;
}


// draw circles on the desired image
void drawCricles(Mat img, vector<Vec3f> circles) {
    for (size_t i = 0; i<circles.size(); i++)
    {
        // circle center and radius
        Point center = Point(circles[i][0], circles[i][1]);
        int radius = circles[i][2];
        // circle outline
        circle(img, center, 1, Scalar(255,0,255), 3, LINE_AA);
        circle(img, center, radius, Scalar(255,0,255), 2, LINE_AA);
    }
}


// extract a possible coin from the circle data
Coin extractCoin(Mat input_img, Vec3f circle){
    // Check for out-of-bound errors
    int x = circle[0]-circle[2];
    if (x < 0){
        x = 0;
    }
    int y = circle[1]-circle[2];
    if (y < 0){
        y = 0;
    }
    int w = 2*circle[2];
    if (x + w > input_img.cols){
        w = input_img.cols - x;
    }
    int h = 2*circle[2];
    if (y + h > input_img.rows){
        h = input_img.rows - y;
    }
    // Get the image portion containing the circle
    Rect roi = Rect(x,y,w,h);
    Mat square = input_img(roi);
    // Mask the circle
    Mat mask = Mat::zeros(square.size(), CV_8U);
    int r = (square.cols/2);
    cv::circle(mask, Point(r,r), r, Scalar(255), -1);
    Mat coin_img = Mat::zeros(square.size(), square.type());
    square.copyTo(coin_img, mask);
    /*
     namedWindow("test");
     imshow("test", coin_img);
     cv::waitKey(0);
     */
    // Construct coin
    return Coin(coin_img);
}

