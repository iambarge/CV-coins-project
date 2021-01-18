//
//  coins_toolbox.hpp
//  coins_toolbox
//
//  Created by Nicolò Bargellesi on 27/06/2019.
//
//

#ifndef coins_toolbox_hpp
#define coins_toolbox_hpp

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Feature Parameters Data Structure
struct FParams {
    std::string type;               // ORB / SIFT / SURF
    
    // ORB parameters
    int nfeatures;                  //  maximum number of features to retain
    float scaleFactor;              //  pyramid decimation ratio
    int nlevels;                    //  number of pyramid levels
    int edgeThreshold;              //  border where the features are not detected
    int firstLevel;                 //  level of pyramid to put source image to
    int WTA_K;                      //  number of points that produce each element of the oriented BRIEF descriptor
    int patchSize;                  //  size of the patch used by the oriented BRIEF descriptor
    int fastThreshold;              //  fast threshold
    
    // SIFT parameters
    //int nfeatures;                //  maximum number of features to retain
    int nOctaveLayers;              //  number of layers in each octave
    double contrastThreshold;       //  contrast threshold used to filter out weak features
    //double edgeThreshold;         //  threshold used to filter out edge-like features
    double sigma;                   //  sigma of the Gaussian applied to the input image at the octave #0
    
    // SURF parameters
    double hessianThreshold;        //  threshold for hessian keypoint detector
    int nOctaves;                   //  number of pyramid octaves
    //int nOctaveLayers;            //  number of layers in each octave
    bool extended;                  //  Extended descriptor flag
    bool upright;                   //  Up-right or Rotated features flag
};


// Class for the coin object
class Coin{
    
    // Methods
    
public:
    
    // constructors
    
    // unknown value constructor
    Coin(cv::Mat coin_img);
    
    // known value constructor
    Coin(cv::Mat coin_img, float coin_val);
    
    // get/set methods
    cv::Mat getImage();
    
    float getValue();
    
    void setValue(float val);
    
    cv::Mat getDescriptor();
    
    std::vector<cv::KeyPoint> getKeypoints();
    
    // detects 1 and 2 Euro coins via HSV Saturation color matching
    //  reference:       reference labeled coin
    //  colorThreshold:  threshold value for the match
    //  OUTPUT: 1 -> match with reference
    //          0 -> NOT match
    bool colorMatchEuro(Coin reference, int colorThreshold);
    
    // detects copper coins via Lab a-component color matching
    //  colorThreshold:  threshold value for the match
    //  OUTPUT: 1 -> copper
    //          0 -> other
    bool colorMatchCent(int colorThreshold);
    
    // extracts ORB features from coin image
    // input:   ORB::create parameters
    void ORBfeatures(FParams feat_params);
    
    // extracts SIFT features from coin image
    // input:   SIFT::create parameters
    void SIFTfeatures(FParams feat_params);
    
    // extracts SURF features from coin image
    // input:   SURF::create parameters
    void SURFfeatures(FParams feat_params);
    
    // computes matches between the features extracted from the coin and a reference coin
    //  reference:      reference labeled coin
    //  max_distance:   select the matches with distance less than max_distance
    //  RANSAC_th:      maximum allowed reprojection error to treat a point pair as an inlier
    //  feat_type:      ORB / SIFT / SURF
    //  OUTPUT:         match score (in terms of number of matches)
    long patternMatch(Coin reference, int max_distance, int RANSAC_th, std::string feat_type);
    
    // guesses the value of the coin
    //  dataset:        dataset of labeled coins
    //  max_distance:   select the matches with distance less than max_distance
    //  RANSAC_th:      maximum allowed reprojection error to treat a point pair as an inlier
    //  uncertainty_th: "NO COIN detected" threshold
    //  colorThreshold: threshold value for color matching
    //  feat_params:    whole set of feature parameters
    //  OUTPUT:         guessed value
    float guessValue(std::vector<Coin> dataset, int max_distance, int RANSAC_th, long uncertainty_th, int color_threshold,  FParams feat_params);
    
protected:
    
    // Data
    
    // coin image
    cv::Mat coin_image;
    
    // coin value
    float coin_value;
    
    // coin image keypoints
    std::vector<cv::KeyPoint> keypoints;
    
    // coin image descriptor
    cv::Mat descriptor;
    
};

#endif /* coins_toolbox_hpp */
