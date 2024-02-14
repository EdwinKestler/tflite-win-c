// tflite-c-windows.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <chrono>
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ObjectDetector.h"

using namespace cv;
using namespace std;


int main()
{
	VideoCapture cap(0);
	ObjectDetector detector = ObjectDetector("ssd_mobilenet_v3_float.tflite", false, false);
	int i = 0;
	long long duration = 0;
	double fps = 0;
	while (true) {
		Mat frame;
		cap >> frame;

		auto start = chrono::high_resolution_clock::now();
		DetectResult* res = detector.detect(frame);
		auto stop = chrono::high_resolution_clock::now();
		for (int i = 0; i < detector.DETECT_NUM; ++i) {
			int label = res[i].label;
			float score = res[i].score;
			float xmin = res[i].xmin;
			float xmax = res[i].xmax;
			float ymin = res[i].ymin;
			float ymax = res[i].ymax;

			rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
			putText(frame, to_string(label) + "-" + to_string(score), Point(xmin, ymin), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255), 2);
		}

		auto d = chrono::duration_cast<chrono::milliseconds>(stop - start);
		duration += d.count();
		if (++i % 5 == 0) {
			fps = (1000.0 / duration) * 5;
			duration = 0;
		}

		putText(frame, to_string((int)fps) + " fps", Point(20, 20), FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 0), 2);

		imshow("frame", frame);
		int k = waitKey(50);
		if (k > 0) {
			break;
		}
	}
}