//
// Copyright (c) 2019, Nicolò Bargellesi
//
// This source code is licensed under the MIT-style license found in the
// LICENSE file in the root directory of this source tree.
//

#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ccalib.hpp>
#include "coins_toolbox.hpp"

using namespace std;
using namespace cv;

// unknown value constructor
Coin::Coin(Mat coin_img) {
    coin_image = coin_img;
}


// known value constructor
Coin::Coin(Mat coin_img, float coin_val) {
    coin_image = coin_img;
    coin_value = coin_val;
}


// get/set methods
cv::Mat Coin::getImage(){
    return coin_image;
}

float Coin::getValue(){
    return coin_value;
}

cv::Mat Coin::getDescriptor(){
    return descriptor;
}

vector<KeyPoint> Coin::getKeypoints(){
    return keypoints;
}

void Coin::setValue(float val){
    coin_value = val;
}


// detects 1 and 2 Euro coins via HSV Saturation color matching
bool Coin::colorMatchEuro(Coin reference, int colorThreshold){
    Mat color, colorChannels[3], ref_color, ref_colorChannels[3];
    cvtColor(this->coin_image, color, COLOR_BGR2HSV);          // color conversion
    cvtColor(reference.getImage(), ref_color, COLOR_BGR2HSV);  // color conversion
    split(color, colorChannels);                               // channels division
    split(ref_color, ref_colorChannels);                       // channels division
    
    // Match sizes
    resize(ref_colorChannels[1], ref_colorChannels[1], Size(color.size().width, color.size().height));
    
    // binary threshold the Saturation channel
    threshold(colorChannels[1], colorChannels[1], 0, 255, THRESH_OTSU); // THRESH_BINARY (40)
    threshold(ref_colorChannels[1], ref_colorChannels[1], 0, 255, THRESH_OTSU); // THRESH_BINARY (50)
    /*
     namedWindow("test");
     imshow("test", ref_colorChannels[1]);
     cv::waitKey(0);
     */
    // XOR
    Mat sub;
    bitwise_xor(colorChannels[1], ref_colorChannels[1], sub);
    /*
     cout << "\n" << mean(sub)[0];
     namedWindow("test");
     imshow("test", sub);
     cv::waitKey(0);
     */
    if (mean(sub)[0] < colorThreshold)
        return 1;       // match
    else
        return 0;       // NO match
}


// detects copper coins via Lab a-component color matching
bool Coin::colorMatchCent(int colorThreshold){
    Mat color, colorChannels[3], ref_color, ref_colorChannels[3];
    cvtColor(this->coin_image, color, COLOR_BGR2Lab);          // color conversion
    split(color, colorChannels);                               // channels division
    /*
     namedWindow("test");
     imshow("test", colorChannels[1]);
     cv::waitKey(0);
     */
    
    // binary threshold the a-component channel
    threshold(colorChannels[1], colorChannels[1], 135, 255, THRESH_BINARY); 
    /*
     cout << "\n" << mean(colorChannels[1])[0];
     namedWindow("test");
     imshow("test", colorChannels[1]);
     cv::waitKey(0);
     */
    if (mean(colorChannels[1])[0] > colorThreshold)
        return 1;
    else
        return 0;
}


// extracts ORB features from coin image
void Coin::ORBfeatures(FParams feat_params) {
    // extract desired parameters
    int nfeatures = feat_params.nfeatures;
    float scaleFactor = feat_params.scaleFactor;
    int nlevels = feat_params.nlevels;
    int edgeThreshold = feat_params.edgeThreshold;
    int firstLevel = feat_params.firstLevel;
    int WTA_K = feat_params.WTA_K;
    int patchSize = feat_params.patchSize;
    int fastThreshold = feat_params.fastThreshold;
    
    vector<KeyPoint> keys;
    Mat desc;
    Mat detect_region = coin_image;
    Ptr<ORB> detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize, fastThreshold);
    
    // mask the center (70% radius)
    Mat mask = Mat::zeros(coin_image.size(), CV_8U);
    int c = (coin_image.cols/2);
    int r = (coin_image.cols/2)*70/100;
    circle(mask, Point(c,c), r, Scalar(255), -1);
    
    detector->detectAndCompute(detect_region, mask, keys, desc, false);
    
    // Show detected features
    /*
        Mat output_img;
        drawKeypoints(coin_image, keys, output_img, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        imshow("orb_mwe", output_img);
        waitKey(0);
    */
    //
    
    keypoints = keys;
    descriptor = desc;
}


// extracts SIFT features from coin image
void Coin::SIFTfeatures(FParams feat_params) {
    // extract desired parameters
    int nfeatures = feat_params.nfeatures;
    int nOctaveLayers = feat_params.nOctaveLayers;
    double contrastThreshold = feat_params.contrastThreshold;
    double edgeThreshold = feat_params.edgeThreshold;
    double sigma = feat_params.sigma;
    
    vector<KeyPoint> keys;
    Mat desc;
    Mat detect_region = coin_image;
    Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);
    
    // mask the center (70% radius)
    Mat mask = Mat::zeros(coin_image.size(), CV_8U);
    int c = (coin_image.cols/2);
    int r = (coin_image.cols/2)*70/100;
    circle(mask, Point(c,c), r, Scalar(255), -1);
    
    detector->detectAndCompute(detect_region, mask, keys, desc, false);
    
    // Show detected features
    /*
    Mat output_img;
    drawKeypoints(coin_image, keys, output_img, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("orb_mwe", output_img);
    waitKey(0);
    */
    //
    
    keypoints = keys;
    descriptor = desc;
}


// extracts SURF features from coin image
void Coin::SURFfeatures(FParams feat_params) {
    // extract desired parameters
    double hessianThreshold = feat_params.hessianThreshold;
    int nOctaves = feat_params.nOctaves;
    int nOctaveLayers = feat_params.nOctaveLayers;
    bool extended = feat_params.extended;
    bool upright = feat_params.upright;
    
    vector<KeyPoint> keys;
    Mat desc;
    Mat detect_region = coin_image;
    Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(hessianThreshold,nOctaves,nOctaveLayers,extended,upright);
    
    // mask the center (70% radius)
    Mat mask = Mat::zeros(coin_image.size(), CV_8U);
    int c = (coin_image.cols/2);
    int r = (coin_image.cols/2)*70/100;
    circle(mask, Point(c,c), r, Scalar(255), -1);
    
    detector->detectAndCompute(detect_region, mask, keys, desc, false);
    
    // Show detected features
    /*
    Mat output_img;
    drawKeypoints(coin_image, keys, output_img, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("orb_mwe", output_img);
    waitKey(0);
    */
    //
    
    keypoints = keys;
    descriptor = desc;
}


// computes matches between the features extracted from the coin and a reference coin
long Coin::patternMatch(Coin reference, int max_distance, int RANSAC_th, string feat_type) {
    vector<DMatch> matches, ref_matches, final_matches;
    Ptr<BFMatcher> matcher;
    // select the correct distance
    if(feat_type.compare("ORB")==0){
        matcher = BFMatcher::create(NORM_HAMMING, true);
    }
    else {
        matcher = BFMatcher::create(NORM_L2, true);
    }
    // find and refine matches
    matcher->match(reference.getDescriptor(), this->descriptor, matches);
    
    for (int j=0; j < matches.size(); j++) {
        if (abs(matches[j].distance) <= max_distance){
            ref_matches.push_back(matches[j]);
        }
    }
    // Show detected matches
    /*
        Mat img_matches;
        drawMatches(reference.getImage(), reference.getKeypoints(), this->coin_image, this->keypoints,
                    ref_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>()); //DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
        imshow( "Matches", img_matches );
        waitKey(0);
    */
    //
    
    if(ref_matches.size()>0) {      //  if matches are detected
        // Localize the objects
        vector<Point2f> obj, scene;
        Mat mask;
    
        for(int i = 0; i < ref_matches.size(); i++ )
        {
            // Get the keypoints from the matches
            obj.push_back(reference.getKeypoints()[ref_matches[i].queryIdx].pt);
            scene.push_back(keypoints[ref_matches[i].trainIdx].pt);
        }
        // Refine the matches through RANSAC
        Mat H = findHomography(obj, scene, RANSAC, RANSAC_th, mask);
        for(int i = 0; i < ref_matches.size(); i++ )
        {
            if((unsigned int)mask.at<uchar>(i)) {
                final_matches.push_back(ref_matches[i]);
            }
        }
        // Show detected matches
        /*
         //cout << "\n" << ref_matches.size() << " - " << ref_obj.size();
         drawMatches(reference.getImage(), reference.getKeypoints(), this->coin_image, this->keypoints,
         final_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
         vector<char>()); //DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
         imshow( "Matches", img_matches );
         waitKey(0);
        */
        //
        
        // Return # of matches
        return final_matches.size();            //  # of matches
    }
    else {
        return 0;
    }
}


// guesses the value of the coin
float Coin::guessValue(vector<Coin> dataset, int max_distance, int RANSAC_th, long uncertainty_th, int color_threshold,  FParams feat_params) {
    // COLOR MATCH (for 1 and 2 Euros)
    for(int i=0; i<2; i++){
        if(this->colorMatchEuro(dataset[i], color_threshold)){
            return dataset[i].getValue();
        }
    }
    
    // PATTERN MATCH (for others)
    string feat_type = feat_params.type;
    if(feat_type.compare("ORB")==0) {
        // Update to match dataset images size
        feat_params.edgeThreshold = this->coin_image.rows*feat_params.edgeThreshold/600;
        feat_params.patchSize = this->coin_image.rows*feat_params.patchSize/600;
        this->ORBfeatures(feat_params);
    } else if (feat_type.compare("SIFT")==0) {
        this->SIFTfeatures(feat_params);
    } else if (feat_type.compare("SURF")==0) {
        this->SURFfeatures(feat_params);
    } else {
        std::cerr << feat_type << " feature type NOT FOUND (it should be ORB / SIFT / SURF). \n";
        throw runtime_error("Error extracting features");
    }
    
    // COLOR MATCH (for cents)
    long max_score = 0;
    float guess = -1;
    if(this->colorMatchCent(color_threshold)){      // if copper
        // Only 1,2,5 cents check
        for(int i=5; i<dataset.size(); i++){
            long res = this->patternMatch(dataset[i], max_distance, RANSAC_th, feat_type);
            if( res > max_score && res > uncertainty_th) {
                max_score = res;
                guess = dataset[i].getValue();
            }
        }
        return guess;
    }
    else {
        // Only 10,20,50 cents check
        for(int i=2; i<5; i++){
            long res = this->patternMatch(dataset[i], max_distance, RANSAC_th, feat_type);
            if( res > max_score && res > uncertainty_th) {
                max_score = res;
                guess = dataset[i].getValue();
            }
        }
        return guess;
    }
}
