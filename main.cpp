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

// Function to compute gradients in x and y directions using Sobel operators
void computeGradient(const Mat& img, Mat& gradX, Mat& gradY) {
    Sobel(img, gradX, CV_32F, 1, 0, 3);
    Sobel(img, gradY, CV_32F, 0, 1, 3);
}

// Function to compute Lucas-Kanade optical flow for a given window
void LucasKanadeOpticalFlow(const Mat& prevImg, const Mat& nextImg, Mat& flowX, Mat& flowY, int windowSize = 3) {
    Mat gradX, gradY;
    computeGradient(prevImg, gradX, gradY);

    flowX = Mat(prevImg.size(), CV_32F);
    flowY = Mat(prevImg.size(), CV_32F);

    int halfWindowSize = windowSize / 2;

    for (int y = halfWindowSize; y < prevImg.rows - halfWindowSize; ++y) {
        for (int x = halfWindowSize; x < prevImg.cols - halfWindowSize; ++x) {
            float A[4] = { 0.0f };
            float b[2] = { 0.0f };

            for (int j = -halfWindowSize; j <= halfWindowSize; ++j) {
                for (int i = -halfWindowSize; i <= halfWindowSize; ++i) {
                    float Ix = gradX.at<float>(y + j, x + i);
                    float Iy = gradY.at<float>(y + j, x + i);
                    float It = static_cast<float>(nextImg.at<uchar>(y + j, x + i)) - static_cast<float>(prevImg.at<uchar>(y + j, x + i));

                    A[0] += Ix * Ix;
                    A[1] += Ix * Iy;
                    A[2] += Ix * Iy;
                    A[3] += Iy * Iy;

                    b[0] += -Ix * It;
                    b[1] += -Iy * It;
                }
            }

            float detA = A[0] * A[3] - A[1] * A[2];
            if (std::abs(detA) > 1e-5) {
                float invDetA = 1.0f / detA;
                flowX.at<float>(y, x) = (A[3] * b[0] - A[1] * b[1]) * invDetA;
                flowY.at<float>(y, x) = (A[0] * b[1] - A[2] * b[0]) * invDetA;
            }
        }
    }
}

void createImagePyramid(const Mat& img, vector<Mat>& pyramid, int levels) {
    pyramid.push_back(img);

    for (int l = 1; l < levels; ++l) {
        Mat downsampledImg;
        pyrDown(pyramid[l - 1], downsampledImg);
        pyramid.push_back(downsampledImg);
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
    int codec = VideoWriter::fourcc('x', '2', '6', '4');
    outputVideo.open(outputFilename, codec, fps, Size(frameWidth, frameHeight));

    if (!outputVideo.isOpened()) {
        cerr << "Could not create the output video file." << endl;
        return;
    }

    Mat frame, prevFrame, gray, prevGray;
    Mat flowX, flowY;
    cap>> prevFrame;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Create image pyramids for current and previous frames
        vector<Mat> prevPyr, currPyr;
        createImagePyramid(prevGray, prevPyr, 3);
        createImagePyramid(gray, currPyr, 3);

        // Perform Lucas-Kanade optical flow between consecutive frames using pyramids
        LucasKanadeOpticalFlow(prevPyr[0], currPyr[0], flowX, flowY);

        // Visualize optical flow as arrows on the frame
        Mat flowVis;
        cvtColor(prevFrame, flowVis, COLOR_GRAY2BGR);

        if (!prevFrame.empty()) {
            cvtColor(prevFrame, flowVis, COLOR_GRAY2BGR); // Convert previous frame to BGR for visualization
        }

        for (int y = 0; y < frame.rows; y += 10) {
            for (int x = 0; x < frame.cols; x += 10) {
                Point2f flow = Point2f(flowX.at<float>(y, x), flowY.at<float>(y, x));
                line(flowVis, Point(x, y), Point(cvRound(x + flow.x), cvRound(y + flow.y)), Scalar(0, 255, 0));
                circle(flowVis, Point(cvRound(x + flow.x), cvRound(y + flow.y)), 1, Scalar(0, 0, 255), -1);
            }
        }

        imshow("Pyramid KLT Tracking", flowVis);

        outputVideo.write(flowVis); // Write frame with optical flow visualization to video

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