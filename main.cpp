#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

const int fast_threshold = 20; 

struct FeatureTrack {
    Point2f point;
    vector<Point2f> track;
};


void PyramidKLT(string path) {
    VideoCapture video(path);

    // Check if the video is opened successfully
    if (!video.isOpened()) {
        cerr << "Could not open the video." << endl;
        return;
    }

    // Video properties
    int frameWidth = static_cast<int>(video.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(video.get(CAP_PROP_FRAME_HEIGHT));
    double fps = video.get(CAP_PROP_FPS);

    // Define the codec and create VideoWriter object
    VideoWriter outputVideo;
    string outputFilename = "/results/PyramidKLT.mp4"; // Change this filename as needed
    int codec = VideoWriter::fourcc('x', 'P', '4', 'V');
    outputVideo.open(outputFilename, codec, fps, Size(frameWidth, frameHeight));

    if (!outputVideo.isOpened()) {
        cerr << "Could not create the output video file." << endl;
        return;
    }

    // Read the first frame
    Mat prevFrame, prevGray;
    video >> prevFrame;

    // Convert the first frame to grayscale
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    // Feature detection using goodFeaturesToTrack
    vector<Point2f> keypoints;
    int maxCorners = 1000;
    double qualityLevel = 0.01;
    double minDistance = 10.0;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    goodFeaturesToTrack(prevGray, keypoints, maxCorners, qualityLevel, minDistance,
                        Mat(), blockSize, useHarrisDetector, k);

    // Initialize FeatureTrack objects for detected keypoints
    vector<FeatureTrack> featureTracks;
    for (const auto& keypoint : keypoints) {
        FeatureTrack track;
        track.point = keypoint;
        track.track.push_back(keypoint);
        featureTracks.push_back(track);
    }

    Mat frame, gray;
    vector<Point2f> nextPoints;

    while (true) {
    // Read the next frame
    video >> frame;

    // Check if the video ends
    if (frame.empty()) {
        break;
    }

    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Pyramidal Lucas-Kanade optical flow for feature tracking
    vector<Point2f> prevKeypoints = keypoints;
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

    calcOpticalFlowPyrLK(prevGray, gray, prevKeypoints, nextPoints, status, err,
                         Size(21, 21), 3, criteria);

    // Display the frame with tracked features
    for (size_t i = 0; i < nextPoints.size(); ++i) {
        if (status[i] == 1) {
            line(frame, prevKeypoints[i], nextPoints[i], Scalar(0, 255, 0), 2);
            circle(frame, nextPoints[i], 3, Scalar(0, 0, 255), -1);
        }
    }

    // Display the current frame with tracked features
    imshow("Pyramid KLT Tracking", frame);

    // Write the frame with keypoints and tracked points to the output video
    outputVideo.write(frame);


    // Update the previous frame and keypoints for the next iteration
    prevGray = gray.clone();
    keypoints = nextPoints;

    // Exit if the 'Esc' key is pressed
    char key = waitKey(30);
    if (key == 27) {
        break;
    }
}

    // Release resources
    video.release();
    outputVideo.release();
    destroyAllWindows();
}


void LucasKanadeOpticalFlow(const vector<Point2f>& prevCorners, vector<Point2f>& newCorners, Mat& Ix, Mat& Iy, Mat& It, int windowSize = 5) {
    Mat u(Ix.size(), CV_32F, Scalar(1.0)); // Flow in the x-direction
    Mat v(Ix.size(), CV_32F, Scalar(1.0)); // Flow in the y-direction
    Mat A = Mat::zeros(2, 2, CV_32F); // Matrix A for Lucas-Kanade
    Mat b = Mat::zeros(2, 1, CV_32F); // Vector b for Lucas-Kanade
    for (size_t i = 0; i < prevCorners.size(); ++i) {
        int y = static_cast<int>(prevCorners[i].y);
        int x = static_cast<int>(prevCorners[i].x);

        // Optical flow calculation for a specific corner point
        Rect roi(x - 2, y - 2, windowSize, windowSize);  // Define ROI for the 5x5 window
        Mat IxROI = Ix(roi);           // Extract 5x5 windows from Ix, Iy, It
        Mat IyROI = Iy(roi);
        Mat ItROI = It(roi);

        A.at<float>(0, 0) = static_cast<float>(sum(IxROI.mul(IxROI))[0]);
        A.at<float>(0, 1) = static_cast<float>(sum(IxROI.mul(IyROI))[0]);
        A.at<float>(1, 0) = A.at<float>(0, 1);
        A.at<float>(1, 1) = static_cast<float>(sum(IyROI.mul(IyROI))[0]);

        b.at<float>(0, 0) = static_cast<float>(-sum(IxROI.mul(ItROI))[0]);
        b.at<float>(1, 0) = static_cast<float>(-sum(IyROI.mul(ItROI))[0]);

        Mat c = A.inv() * b;

        u.at<float>(y, x) = c.at<float>(0, 0);
        v.at<float>(y, x) = c.at<float>(1, 0);

        newCorners[i] = Point2f(x + u.at<float>(y, x), y + v.at<float>(y, x));
    }
}


void MyPyramidKLT(string path)
{
    VideoCapture cap(path);

    // Check if the video is opened successfully
    if (!cap.isOpened()) {
        cerr << "Could not open the video." << endl;
        return;
    }

    // Video properties
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);

    // Define the codec and create VideoWriter object
    VideoWriter outputVideo;
    string outputFilename = "C:/work/results/pyramid.mp4"; // Change this filename as needed
    int codec = VideoWriter::fourcc('x', 'P', '4', 'V');
    outputVideo.open(outputFilename, codec, fps, Size(frameWidth, frameHeight));

    if (!outputVideo.isOpened()) {
        cerr << "Could not create the output video file." << endl;
        return;
    }

    Mat frame, prevFrame, gray, prevGray, img;
    cap>> prevFrame;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);
    prevGray.convertTo(prevGray, CV_32F, 1.0 / 255.0);

    vector<Point2f> keypoints;
    int maxCorners = 100;
    double qualityLevel = 0.01;
    double minDistance = 10.0;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    goodFeaturesToTrack(prevGray, keypoints, maxCorners, qualityLevel, minDistance,
                        Mat(), blockSize, useHarrisDetector, k);

    Mat Gx = (Mat_<float>(2, 2) << -1, 1, -1, 1);
    Mat Gy = (Mat_<float>(2, 2) << -1, -1, 1, 1);
    Mat Gt1 = (Mat_<float>(2, 2) << -1, -1, -1, -1);
    Mat Gt2 = (Mat_<float>(2, 2) << 1, 1, 1, 1);
    vector<Point2f> nextKeypoints;

    while (true) {
        cap >> frame;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F, 1.0 / 255.0);
        

        Mat Ix, Iy, It, It1, It2;
        Ix.convertTo(Ix, CV_32F);
        Iy.convertTo(Iy, CV_32F);
        It.convertTo(It, CV_32F);
        It1.convertTo(It1, CV_32F);
        It2.convertTo(It2, CV_32F);
        filter2D(prevGray, Ix, -1, Gx);
        filter2D(gray, Iy, -1, Gy);
        filter2D(prevGray, It1, -1, Gt1);
        filter2D(gray, It2, -1, Gt2);
        It = It1 + It2;
        
        nextKeypoints = keypoints;

        LucasKanadeOpticalFlow(keypoints, nextKeypoints, Ix, Iy, It);

        for (size_t i = 0; i < keypoints.size(); ++i) {
            line(frame, keypoints[i], nextKeypoints[i], Scalar(0, 255, 0), 2);
            circle(frame, nextKeypoints[i], 3, Scalar(0, 0, 255), -1);
        }

        imshow("Pyramid KLT Tracking", frame);

        outputVideo.write(frame); // Write frame with optical flow visualization to video

        // Update previous frame and previous grayscale frame
        frame.copyTo(prevFrame);
        gray.copyTo(prevGray);
        

        if (waitKey(1) == 27) {
        break; // Exit loop if ESC is pressed
        }
    }

    cap.release();
    outputVideo.release();
    destroyAllWindows();
}

int main(int, char**){

    string path = "C:/work/KLT/4.gif";
    //PyramidKLT(path);
    MyPyramidKLT(path);

    return 0;
}