#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "ardrone/ardrone.h"
#include <pthread.h>
#include <string>
#include <sstream>
#include <time.h>
#include <fstream>
#include <ctime>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;

#define NUM_THREADS 1

ARDrone ardrone;
int FLAG = 0;
struct timespec start, finish;
double elapsed;

/* Used for FPS */
int CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

/* Used for FPS */
double _avgdur=0;
int _fpsstart=0;
double _avgfps=0;
double _fps1sec=0;

/* Used for FPS */
double avgfps()
{
    if(CLOCK()-_fpsstart>1000)      
    {
        _fpsstart=CLOCK();
        _avgfps=0.7*_avgfps+0.3*_fps1sec;
        _fps1sec=0;
    }

    _fps1sec++;
    return _avgfps;
}


/*Used for segmentation */
Scalar hsv_to_rgb(Scalar c) {
    Mat in(1, 1, CV_32FC3);
    Mat out(1, 1, CV_32FC3);

    float * p = in.ptr<float>(0);

    p[0] = (float)c[0] * 360.0f;
    p[1] = (float)c[1];
    p[2] = (float)c[2];

    cv::cvtColor(in, out, cv::COLOR_HSV2RGB);

    Scalar t;

    Vec3f p2 = out.at<Vec3f>(0, 0);

    t[0] = (int)(p2[0] * 255);
    t[1] = (int)(p2[1] * 255);
    t[2] = (int)(p2[2] * 255);

    return t;

}

/* Used for segmentation */
Scalar color_mapping(int segment_id) {

    double base = (double)(segment_id) * 0.618033988749895 + 0.24443434;

    return hsv_to_rgb(Scalar(fmod(base, 1.2), 0.95, 0.80));

}

// Struct used in clustering section
typedef struct{
	bool is_grouped;
	cv::Point Pt;
	int ID;
}node;

// Struct used to pass arguments to Pthread
typedef struct{
	double vxt;
	double vyt;
	double vzt;
	double vrt;
}thread_data;

thread_data thread_data_array[NUM_THREADS];

/* Helper functions for object detection. */
void getFeatures(SURF_CUDA, Mat, Mat&, vector<KeyPoint>&);
void getMatches(Mat, Mat, vector<KeyPoint>, vector<KeyPoint>, Ptr<cv::cuda::DescriptorMatcher>, vector<DMatch>&);
void filter(vector<DMatch>, vector<DMatch>&, vector<KeyPoint>, vector<KeyPoint>);
void getGroups(Mat, vector<vector<Point> >&, vector<vector<Point> >&, vector<Point>&, vector<Point>&, vector<KeyPoint>, vector<KeyPoint>, vector<DMatch>);
void getSegmentation(Ptr<cv::ximgproc::segmentation::GraphSegmentation>, Mat, Mat, Mat&, Mat&);

/* Helper function for object avoidance. */
void getMovement(Mat&, Mat, Mat, vector<Point>, vector<Point> , double&, double&, double&, double&);

/* Pthread function for object avoidance. */
/* This Pthread function recieves the x, y, z, and rotation parameters of UAV
 * and moves the UAV accordingly, for 5 iterations.
 * @param threadarg struct used to pass x, y, z, and rotation  parameters by value
*/
void *move(void *threadarg)
{
   thread_data *my_data;

	my_data = (thread_data *) threadarg;


	double vx;
	double vy;
	double vz;

	ardrone.getVelocity(&vx, &vy, &vz);

	for(int i = 0; i < 5; i++)
		ardrone.move3D(0.1, my_data->vyt, my_data->vzt, my_data->vrt);

	pthread_exit(NULL);

}


int main(int argc, char* argv[])
{


	try
	{

		if (!ardrone.open())
			return -1;

		/* Open file for input */
		ofstream pic_file;
		pic_file.open("./flight_data/flight_metric.txt");
		pic_file << " Test data\n";
		
		/* Used to store FPS and write to file */
		double storage = 0;

		cv::Mat frame, currFrame, origFrame, prevFrame, h_currDescriptors, h_prevDescriptors, image_gray, canvas;
		std::vector<cv::KeyPoint> h_currKeypoints, h_prevKeypoints;

		GpuMat d_frame, d_fgFrame, d_greyFrame, prevDescriptors, currDescriptors, d_keypoints;
		GpuMat d_threshold;

		/* Segmentation variable */
		Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg = cv::ximgproc::segmentation::createGraphSegmentation(0.5, 500, 50);

		SURF_CUDA detector(800);
		Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(detector.defaultNorm());

		vector<cv::DMatch> matches, good_matches;

		origFrame = ardrone.getImage();
		cv::resize(origFrame, prevFrame, cv::Size(200,200));

		/* - - - - Preprocess Image - - - - */
//		cv::blur(origFrame, origFrame, cv::Size(3,3));
//		cv::boxFilter(origFrame, origFrame, -1, cv::Size(3,3));
//		cv::medianBlur(origFrame, origFrame, 3);

		getFeatures(detector, prevFrame, h_prevDescriptors, h_prevKeypoints);

		for (;;)
		{

			std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]\r" << std::flush;
			origFrame = ardrone.getImage();
			cv::resize(origFrame, currFrame, cv::Size(200,200));

			cv::imshow("Original", currFrame);
		
			/* - - - - Edge Detection - - - - */
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;

			cv::cvtColor(origFrame, canvas, CV_RGB2GRAY);
			Canny(canvas, canvas, 260, 300, 3);
			cv::imshow("Canny", canvas);

			/* -------Begining of Line Section-------*/
		
			Mat grey_currFrame(canvas);

			vector<Vec4i> currlines;
			HoughLinesP( grey_currFrame, currlines, 1, CV_PI/180, 80, 30, 10 );
			for( size_t i = 0; i < currlines.size(); i++ )
			{
			  line(grey_currFrame, Point(currlines[i][0], currlines[i][1]),
				  Point(currlines[i][2], currlines[i][3]), Scalar(0,0,255), 3, 8);
			}
		
			cv::imshow("Lines", grey_currFrame);
			/* ------------End of Line-------------*/


			dilate(canvas, canvas, cv::Mat(), cv::Point(-1,-1), 2);
			cv::imshow("Dilated", canvas);

			findContours(canvas, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

			canvas = Mat::zeros(canvas.size(), CV_8UC3);
			for(int i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(220,200,20);
				drawContours(canvas, contours, i , color, -10, 8, hierarchy, 0, Point());
			}

		



			/* - - - - Preprocess Image - - - - */
/*			cv::Mat lab_image;
			cv::cvtColor(currFrame, lab_image, CV_BGR2Lab);

			std::vector<cv::Mat> lab_planes(3);
			cv::split(lab_image, lab_planes);

			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
			clahe->setClipLimit(4);
			cv::Mat dst;
			clahe->apply(lab_planes[0], dst);

			dst.copyTo(lab_planes[0]);
			cv::merge(lab_planes, lab_image);

			cv::cvtColor(lab_image, currFrame, CV_Lab2BGR);
*/

//			cv::blur(origFrame, origFrame, cv::Size(3,3));
//			cv::boxFilter(origFrame, origFrame, -1, cv::Size(3,3));
//			cv::medianBlur(origFrame, origFrame, 3);

			clock_gettime(CLOCK_MONOTONIC, &start);
			getFeatures(detector, currFrame, h_currDescriptors, h_currKeypoints);

			vector<DMatch> good_matches;
			getMatches(h_prevDescriptors, h_currDescriptors, h_prevKeypoints, h_currKeypoints, matcher, good_matches);
			currFrame.copyTo(frame);
			try
			{
				if (good_matches.size() > 2)
				{

					Mat prev_output_image, curr_output_image;

					getSegmentation(seg, prevFrame, currFrame, prev_output_image, curr_output_image);
					imshow("curr", curr_output_image);

					vector<vector<Point> > prev_group_pts;
					vector<vector<Point> > curr_group_pts;
					vector<Point> prev_midpoints;
					vector<Point> curr_midpoints;
					getGroups(curr_output_image, prev_group_pts, curr_group_pts, prev_midpoints, curr_midpoints, h_prevKeypoints, h_currKeypoints, good_matches);

					double vx = 1.0, vy = 0.0, vz = 0.0, vr = 0.0;
					getMovement(frame, prev_output_image, curr_output_image, prev_midpoints, curr_midpoints, vx, vy, vz, vr);

					if(FLAG)
					{
						/* Write pic and directions to file*/

						string path = "./flight_data/Obj_";
						time_t ti = time(0);
						struct tm * now = localtime(&ti);
						std::ostringstream oss;
						oss << path << "_" << now->tm_min << "_" << now->tm_sec << ".jpg\n";
						imwrite(oss.str(), frame);
	
						oss << "Left/Right: " << vy << "\tUp/Down: " << vz << endl;
						oss << "Time: " << elapsed << endl;
						oss << "FPS: " << storage << endl;
						pic_file << oss.str() << endl << endl;

					}

			/* ------------------PTHREAD SECTION ----------------------*/


					pthread_t threads[NUM_THREADS];
					int rc;
					long t;

					for(t = 0; t < NUM_THREADS; t++)
					{

						thread_data_array[t].vxt = vx;
						thread_data_array[t].vyt = vy;
						thread_data_array[t].vzt = vz;
						thread_data_array[t].vrt = vr;

						rc = 	pthread_create(&threads[t], NULL, move,
								(void *) &thread_data_array[t]);
						if(rc){
							cout << "Error creating thread" << endl;
							exit(-1);
						}
					}

			/*------------------END OF PTHREAD-------------------------*/
					curr_group_pts.clear();
					prev_group_pts.clear();

				}

			}
			catch (const cv::Exception& ex)
			{

				continue;

			}

			/* Calculate individual FPS */
			storage = avgfps();

			cv::imshow("Contours", canvas);
//			cv::imshow("Normalized", currFrame);

//			cv::imshow("Result", frame);
			char key = cvWaitKey(1);

			if (key == 27) // Esc key
				break;

			currFrame.copyTo(prevFrame);
			h_prevKeypoints.clear();
			h_prevKeypoints = h_currKeypoints;
			h_currKeypoints.clear();
			h_currDescriptors.copyTo(h_prevDescriptors);

			// Take off / Landing
			if (key == ' ')
			{
				if (ardrone.onGround())
				{
					ardrone.takeoff();
					cout << "Start" << endl;
					// Wait(secs) to stabilize, before commands
					sleep(10);
					cout << "End" << endl;
				}
				else ardrone.landing();
			}

        		// Change camera
        		static int mode = 0;
        		if (key == 'c') ardrone.setCamera(++mode % 4);


		}

		cout << "Average FPS: " << avgfps() << endl;

		pic_file.close();
		ardrone.close();

	}
	catch (const cv::Exception& ex)
	{

		std::cout << "Error: " << ex.what() << std::endl;

	}

	return 0;
}

/**
* Generates key points and descriptors of features in a given frame
* @param detector the GPU SURF wrapper for detecting features in frame
* @param frame the frame features will be extracted from
* @param descriptors matrix of the different characteristics of features
* @param keys the keypoints in the frame matching to features
*/
void getFeatures(SURF_CUDA detector, Mat frame, Mat& descriptors, vector<KeyPoint>& keys)
{

       	GpuMat d_frame, d_keypoints, d_descriptors;

        d_frame.upload(frame);
        cuda::cvtColor(d_frame, d_frame, CV_RGB2GRAY);

        detector(d_frame, GpuMat(), d_keypoints, d_descriptors);
        detector.downloadKeypoints(d_keypoints, keys);
        d_descriptors.download(descriptors);

}

/**
* Matches keypoints from the previous frame to the current frame
* @param prevDescriptors the descriptor matrix for the previous frame's key features
* @param currDescriptors the descriptor matrix for the current frame's key features
* @param prevKeys the keypoints of the previous frame
* @param currKeys the keypoints of the current frame
* @param matcher the wrapper for the type of matcher used for matching the previous and current features
* @param good_matches the container for the matches
*/
void getMatches(Mat prevDescriptors, Mat currDescriptors, vector<KeyPoint> prevKeys, vector<KeyPoint> currKeys,
Ptr<cv::cuda::DescriptorMatcher> matcher, vector<DMatch>& good_matches)
{

	GpuMat d_prevDescriptors, d_currDescriptors;
	d_prevDescriptors.upload(prevDescriptors);
	d_currDescriptors.upload(currDescriptors);

	vector<DMatch> matches;
	try
	{

        	/* Perform the matching and store in temp variable */
        	matcher->match(d_prevDescriptors, d_currDescriptors, matches);

	}
	catch (const cv::Exception& ex) {}

    /* Filter out bad matches */
	filter(matches, good_matches, prevKeys, currKeys);

}

/**
* Removes bad matches
* @param matches the matches to be filtered
* @param good_matches the container for the filtered out matches
* @param prevKeys the keypoints of the previous frame
* @param currKeys the keypoints of the current frame
*/
void filter(vector<DMatch> matches, vector<DMatch>& good_matches, vector<KeyPoint> prevKeys, vector<KeyPoint> currKeys)
{

    /* Thresholds for how much features are allowed to move between frames */
	double max_dist = 0;
	double min_dist = 100;

	for (unsigned int i = 0; i < matches.size(); i++)
	{

		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;

		if (dist > max_dist)
			max_dist = dist;

	}


	for (unsigned int i = 0; i < matches.size(); i++)
	{

        /* Grab index of the keypoints in the previous and current frame */
		int prev = matches[i].queryIdx;
		int curr = matches[i].trainIdx;

        /* Determine if the size of features is expanding between frames */
		double ratio = currKeys[curr].size / prevKeys[prev].size;
		if (matches[i].distance < 3*min_dist  && ratio > 1 )
			good_matches.push_back(matches[i]);

	}

}

/**
* Groups features into clusters
* @param prevGroups the container for the previous frame's groups of clustered features
* @param currGroups the container for the current frame's groups of clustered features
* @param prev_midpoints the container for the center of the clusters in the previous frame
* @param curr_midpoints the container for the center of the clusters in the current frame
* @param h_prevKeypoints the keypoints of the previous frame
* @param h_currKeypoints the keypoints of the current frame
* @param good_matches the matches of features between the previous and current frames
*/
void getGroups(Mat curr_output_image, vector<vector<Point> >& prevGroups, vector<vector<Point> >& currGroups, vector<Point>& prev_midpoints, vector<Point>& curr_midpoints, vector<KeyPoint> h_prevKeypoints, vector<KeyPoint> h_currKeypoints, vector<DMatch> good_matches)
{

	vector<node> prev_quick;
	vector<node> curr_quick;

    /* Grab and store the good matches' keypoints from both frames */
	for (unsigned int i = 0; i < good_matches.size(); i++)
	{

		node prev_node = {false, h_prevKeypoints[good_matches[i].queryIdx].pt, -1};
		node curr_node = {false, h_currKeypoints[good_matches[i].trainIdx].pt, -1};
		prev_quick.push_back(prev_node);
		curr_quick.push_back(curr_node);

	}

	/*-------------- Clustering Section ---------------*/

   /* Set up containers for the previous and current clusters */
	std::vector<cv::Point> prev_cluster;
	std::vector<cv::Point > curr_cluster;


	/* Cluster points into groups by color */
	for(unsigned int i = 0; i < good_matches.size(); i++)
	{

		int FLAG2 = 0;

		for(unsigned int j = 0; j < currGroups.size(); j++)
		{
			if(curr_output_image.at<Vec3b>(h_currKeypoints[good_matches[i].trainIdx].pt) == 
				curr_output_image.at<Vec3b>(currGroups[j][0]))
			{
				currGroups[j].push_back(h_currKeypoints[good_matches[i].trainIdx].pt);
				prevGroups[j].push_back(h_prevKeypoints[good_matches[i].queryIdx].pt);
				FLAG2 = 1;
			}
		}
		
		if( currGroups.size() == 0 || FLAG2 == 0);
		{
			std::vector<cv::Point> prev_cluster;
			std::vector<cv::Point > curr_cluster;
			curr_cluster.push_back(h_currKeypoints[good_matches[i].trainIdx].pt);
			prev_cluster.push_back(h_prevKeypoints[good_matches[i].queryIdx].pt);
			prevGroups.push_back(prev_cluster);
			currGroups.push_back(curr_cluster);
		}
			

	}

	/* Compute midpoint for the cluster */
	for(unsigned int p = 0; p < currGroups.size(); p++)
	{
		double cx = 0, cy = 0, px = 0, py = 0;

		for(unsigned int w = 0; w < currGroups[p].size(); w++)
		{
			cx += currGroups[p][w].x;
			cy += currGroups[p][w].y;

			px += prevGroups[p][w].x;
			py += prevGroups[p][w].y;
		}

		cx /= currGroups[p].size();
		cy /= currGroups[p].size();
		px /= currGroups[p].size();
		py /= currGroups[p].size();

		Point c_mid(cx, cy);
		Point p_mid(px, py);

		/* Add the cluster and midpoint to their respective containers */
		if (currGroups[p].size() > 3)
		{
			curr_midpoints.push_back(c_mid);
			prev_midpoints.push_back(p_mid);
		}

	}

}

/* This function is used to perform graph-cut segmentation on two consecutive frames,
 * the output of which is stored in the address of two Mat variables.
 *
 * @param seg Segmentation function used to segment prevFrame and currFrame.
 * @param prevFrame Mat variable holding previous frame of video.
 * @param currFrame Mat variable holding current frame of video.
 * @param prev_output_image Mat variable holding result of segmentation on prevFrame.
 * @param curr_outpout_image Mat variable holding result of segmentation on currFrame.
*/

void getSegmentation(Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg, Mat prevFrame, Mat currFrame, Mat& prev_output_image, Mat& curr_output_image)
{

	/* -------Begining of Line Section-------*/
/*
	Mat grey_prevFrame, grey_currFrame;

	cv::cvtColor(prevFrame,grey_prevFrame,CV_BGR2GRAY);
	cv::cvtColor(currFrame,grey_currFrame,CV_BGR2GRAY);

	vector<Vec4i> prevlines;
	HoughLinesP( grey_prevFrame, prevlines, 1, CV_PI/180, 80, 30, 10 );
	for( size_t i = 0; i < prevlines.size(); i++ )
	{
	  line(prevFrame, Point(prevlines[i][0], prevlines[i][1]),
		  Point(prevlines[i][2], prevlines[i][3]), Scalar(0,0,255), 3, 8);
	}

	vector<Vec4i> currlines;
	HoughLinesP( grey_currFrame, currlines, 1, CV_PI/180, 80, 30, 10 );
	for( size_t i = 0; i < currlines.size(); i++ )
	{
	  line(currFrame, Point(currlines[i][0], currlines[i][1]),
		  Point(currlines[i][2], currlines[i][3]), Scalar(0,0,255), 3, 8);
	}
*/
	/* ------------End of Line-------------*/

	Mat prev_input, prev_output, curr_input, curr_output;

	// Segmentation of previos frame
	seg->processImage(prevFrame, prev_output);

	double mins, maxs;
	minMaxLoc(prev_output, &mins, &maxs);

	int nb_segs = (int)maxs + 1;

//	std::cout << nb_segs << " segments" << std::endl;

	prev_output_image = Mat::zeros(prev_output.rows, prev_output.cols, CV_8UC3);

	uint* p;
	uchar* p2;

	for (int i = 0; i < prev_output.rows; i++) {

		p = prev_output.ptr<uint>(i);
		p2 = prev_output_image.ptr<uchar>(i);

		for (int j = 0; j < prev_output.cols; j++) {
			Scalar color = color_mapping(p[j]);
			p2[j*3] = (uchar)color[0];
			p2[j*3 + 1] = (uchar)color[1];
			p2[j*3 + 2] = (uchar)color[2];
		}
	}

	// Segmentation of current frame
	seg->processImage(currFrame, curr_output);

	minMaxLoc(curr_output, &mins, &maxs);

	nb_segs = (int)maxs + 1;

//	std::cout << nb_segs << " segments" << std::endl;

	curr_output_image = Mat::zeros(curr_output.rows, curr_output.cols, CV_8UC3);

	uint* q;
	uchar* q2;

	for (int i = 0; i < curr_output.rows; i++) {

		q = curr_output.ptr<uint>(i);
		q2 = curr_output_image.ptr<uchar>(i);

		for (int j = 0; j < curr_output.cols; j++) {
			Scalar color = color_mapping(q[j]);
			q2[j*3] = (uchar)color[0];
			q2[j*3 + 1] = (uchar)color[1];
			q2[j*3 + 2] = (uchar)color[2];
		}
	}

}

/* This function is used to determine the movement of the UAV given two  consecutive
 * segmented images and the midpoints of feature clusters in the original images.
 * The area of expansion is calculated of detected obstacle in segmented frames.
 *
 * @param frame Mat variable of current frame of the video and used to determine
 * dimensions of image and rectangle around detected object is drawn on it.
 * @param prev_output_image Mat variable of previous segmented frame. Flood
 * fill is seeded from prev_midpoints and area of obstacle is calculated.
 * @param curr_output_image Mat variable of current segmented frame. Flood
 * fill is seeded from curr_midpoints and area of obstacle is calculated.
 * @param prev_midpoints Vector containing midpoints of feature clusters in previous frame.
 * @param curr_midpoints Vector containing midpoints of feature clusters in current frame.
 * @param vx Address of double used to return x-direction of UAV.
 * @param vy Address of double used to return y-direction of UAV.
 * @param vz Address of double used to return z-direction of UAV.
 * @param vr Address of double used to return rotation of UAV.
*/
void getMovement(Mat& frame, Mat prev_output_image, Mat curr_output_image, vector<Point> prev_midpoints, vector<Point> curr_midpoints, double& vx, double& vy, double& vz, double& vr)
{
	FLAG = 0;

	for (unsigned int i = 0; i < curr_midpoints.size(); i++)
	{

		int prev_fill = 0, curr_fill = 0;
		Rect boundRect;
		clock_gettime(CLOCK_MONOTONIC, &finish);
		elapsed = (finish.tv_sec - start.tv_sec);
		elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

		prev_fill = floodFill(prev_output_image, prev_midpoints[i], cv::Scalar(250, 250, 250));
		curr_fill = floodFill(curr_output_image, curr_midpoints[i], cv::Scalar(250, 250, 250), &boundRect);
		cv::circle(frame, curr_midpoints[i], 10, cv::Scalar(250, 50, 0), 3);

		// Calculate area of expansion of detected objects
		double ratio = (double)curr_fill / prev_fill;

		if (ratio > 1.2 && ratio < 2)
		{

//			cout << "Ratio: " << ratio << endl;

			FLAG = 1;
			// Used to approximate countours to polygons + get bounding rects

			cv::rectangle(frame, boundRect.tl(), boundRect.br(),
							 cv::Scalar(200,200,0), 2, 8, 0);

			// Check which quadrant(s) rectangle is in
			cv::Point top_left(boundRect.tl().x, boundRect.tl().y);
			cv::Point top_right(boundRect.tl().x + boundRect.width, boundRect.tl().y);
			cv::Point bot_left(boundRect.tl().x, boundRect.tl().y + boundRect.height);
			cv::Point bot_right(boundRect.tl().x + boundRect.width, boundRect.tl().y + boundRect.height);

			vy += min(0.1, 0.01 * (top_left.x - frame.cols / 2.0));
			vy += max(-0.1, 0.01 * (top_right.x - frame.cols / 2.0));

//                      vz -= min(1.0, 0.01 * (top_left.y - frame.rows / 2.0));
//                      vz -= max(-1.0, 0.01 * (bot_left.y - frame.rows / 2.0));
//			cout << "Left/Right: " << vy << " Up/Down: " << vz << endl;
/*
	        	double vx;
     			double vy;
        		double vz;
			double A = 1/ratio;
			
			ardrone.getVelocity(&vx, &vy, &vz);
					
			double dist = 0.1 * (0.001) / (1 - sqrt(A));

			std::ostringstream distance;
        		distance << "Distance: " <<  dist;

			putText(frame, distance.str(), boundRect.tl(), FONT_HERSHEY_DUPLEX, 0.25, cv::Scalar(220,100,10), 1);
			//cout << distance.str() << endl;*/
		}

	}

}
