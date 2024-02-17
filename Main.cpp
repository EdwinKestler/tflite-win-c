// tflite-c-windows.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ObjectDetector.h"
#include "rstpStream.h"

using namespace cv;
using namespace std;

// Thread-safe queue to hold frames
template<typename T>
class SafeQueue {
private:
    queue<T> q;
    mutable mutex m;
    condition_variable c;
    bool finished = false; // Indicates no more items will be added
public:
    SafeQueue() {}

    void finish() {
        unique_lock<mutex> lock(m);
        finished = true;
        c.notify_all(); // Notify all waiting threads to check the condition
    }

    bool isFinished() const {
        return finished && q.empty();
    }

    void push(T value) {
        lock_guard<mutex> lock(m);
        q.push(move(value));
        c.notify_one();
    }

    bool pop(T& value) {
        unique_lock<mutex> lock(m);
        c.wait(lock, [this] { return !q.empty() || finished; });
        if (q.empty()) {
            return false;
        }
        value = move(q.front());
        q.pop();
        return true;
    }
};

void processFrame(ObjectDetector& detector, SafeQueue<Mat>& frames, atomic<bool>& running, int& i, long long& duration, double& fps) {
    Mat frame;
    while (running && frames.pop(frame)) { // Check for running flag and if a frame is available
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(640, 320)); // Adjust according to your model's expected input

        try {
            auto start = chrono::high_resolution_clock::now();
            DetectResult* res = detector.detect(resizedFrame);
            auto stop = chrono::high_resolution_clock::now();

            for (int j = 0; j < detector.DETECT_NUM; ++j) {
                int label = res[j].label;
                float score = res[j].score;
                // Use static_cast<int> to explicitly convert float to int
                int xmin = static_cast<int>(res[j].xmin);
                int xmax = static_cast<int>(res[j].xmax);
                int ymin = static_cast<int>(res[j].ymin);
                int ymax = static_cast<int>(res[j].ymax);

                rectangle(resizedFrame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
                // For text positioning, rounding might be preferred to ensure readability
                putText(resizedFrame, to_string(label) + "-" + to_string(score), Point(xmin, ymin - 10), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 255, 0), 2);
            }

            auto d = chrono::duration_cast<chrono::milliseconds>(stop - start);
            duration += d.count();
            if (++i % 5 == 0) {
                fps = (1000.0 / duration) * 5;
                duration = 0;
            }

            putText(resizedFrame, to_string((int)fps) + " fps", Point(20, 20), FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 0), 2);

            // Display the frame with detections
            imshow("Frame", resizedFrame);
            if (waitKey(1) >= 0) break; // Exit on any key press

            }

        catch (const std::exception& e) {
            cerr << "Error during object detection: " << e.what() << endl;
            running = false; // Stop processing on error
        }

        if (!running) {
            break; // Exit loop if running flag is cleared
        }
    }
}

int main() {
    const string rstp_url = "rtsp://admin:FBx!admin2023@192.168.1.108:554";
    VideoCapture cap(rstp_url);
    if (!cap.isOpened()) {
        cerr << "Error opening video stream" << endl;
        return -1;
    }

    ObjectDetector detector("ssd_mobilenet_v3_float.tflite", false, false);
    SafeQueue<Mat> frames;
    atomic<bool> running(true);
    int frameCounter = 0; // New counter for skipping frames
    int i = 0;
    long long duration = 0;
    double fps = 0;

    // Start processing thread
    thread processingThread(processFrame, ref(detector), ref(frames), ref(running), ref(i), ref(duration), ref(fps));

    while (running) {
        Mat frame;
        
        cap >> frame;
        if (frame.empty()) {
            cerr << "Failed to capture frame" << endl;
            running = false; // Signal threads to stop on capture failure
            break;
        }

        frameCounter++;
        if (frameCounter % 6 != 0) { // Skip processing for 5 frames, process on the 6th
            continue; // Skip the rest of the loop and proceed to the next iteration
        }

        // Reset the counter after processing a frame
        frameCounter = 0;

        frames.push(frame); // Add frame to the queue for processing
    }

    frames.finish(); // Signal no more frames will be added

    processingThread.join(); // Wait for processing thread to finish

    cap.release(); // Explicitly release the VideoCapture resource
    destroyAllWindows(); // Close all OpenCV windows

    return 0;
}

