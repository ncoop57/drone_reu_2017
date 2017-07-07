/*
 * drone.cpp
 *
 *  Created on: Jun 8, 2017
 *      Author: natha
 */

#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ximgproc/segmentation.hpp"
#include <ctime>

using namespace std;
using namespace cv;

Scalar hsv_to_rgb(Scalar c) {
    Mat in(1, 1, CV_32FC3);
    Mat out(1, 1, CV_32FC3);

    float * p = in.ptr<float>(0);

    p[0] = (float)c[0] * 360.0f;
    p[1] = (float)c[1];
    p[2] = (float)c[2];

    cvtColor(in, out, COLOR_HSV2RGB);

    Scalar t;

    Vec3f p2 = out.at<Vec3f>(0, 0);

    t[0] = (int)(p2[0] * 255);
    t[1] = (int)(p2[1] * 255);
    t[2] = (int)(p2[2] * 255);

    return t;

}

Scalar color_mapping(int segment_id) {

    double base = (double)(segment_id) * 0.618033988749895 + 0.24443434;

    return hsv_to_rgb(Scalar(fmod(base, 1.2), 0.95, 0.80));

}

int main()
{

	VideoCapture cap(1);


	Mat frame, output, output_image;

	while (true)
	{

		cap.read(frame);

		resize(frame, output, Size(), 0.50, 0.50);

		Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg = cv::ximgproc::segmentation::createGraphSegmentation(0.5, 1000, 30);

		seg->processImage(output, output);

		double min, max;
		minMaxLoc(output, &min, &max);

		int nb_segs = (int)max + 1;

		std::cout << nb_segs << " segments" << std::endl;

		output_image = Mat::zeros(output.rows, output.cols, CV_8UC3);

		uint* p;
		uchar* p2;

		for (int i = 0; i < output.rows; i++) {

			p = output.ptr<uint>(i);
			p2 = output_image.ptr<uchar>(i);

			for (int j = 0; j < output.cols; j++) {
				Scalar color = color_mapping(p[j]);
				p2[j*3] = (uchar)color[0];
				p2[j*3 + 1] = (uchar)color[1];
				p2[j*3 + 2] = (uchar)color[2];
			}
		}

		imshow("output", output_image);
		imshow("frame", frame);
		char key = waitKey(1);
		if (key == 27)
			break;

	}

}
