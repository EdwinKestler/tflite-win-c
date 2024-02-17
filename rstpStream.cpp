// rstpStream.cpp
#include "rstpStream.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>

void openRTSPStream(const std::string& rstp_url) {
    cv::Mat frame;
    cv::VideoCapture cap;

    int retries = 5;
    for (int i = 0; i < retries; ++i) {
        cap.open(rstp_url, cv::CAP_FFMPEG);
        if (cap.isOpened()) {
            std::cout << "RTSP stream opened successfully." << std::endl;
            break;
        } else {
            std::cerr << "Failed to open RTSP stream. Retrying... (" << i + 1 << "/" << retries << ")" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    if (!cap.isOpened()) {
        std::cerr << "Cannot open RTSP stream after retries." << std::endl;
        return;
    }

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Received an empty frame. Stream may have ended or encountered an error." << std::endl;
            break;
        }

        cv::imshow("RTSP stream", frame);

        if (cv::waitKey(1) == 27) {
            std::cout << "ESC key pressed. Exiting..." << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}