/* Description: Program can detect expanding features 
 *		by drawing a template around features, then checking 
 *		for expansion. If they are expanding, then place in 
 *		vector<KeyPoint> expanding. Avoidance command sent
 *		based off location of features
 * Progress: 
 *		Expansion - 95% Complete (Parameters need fine tuning)
 *		Other (i.e Flood fill) - Not Started
 *		Avoidance - Not started.
*/

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
void getTemplate(Mat, Mat, vector<KeyPoint>, vector<KeyPoint>, vector<DMatch>&, vector<KeyPoint>&, vector<KeyPoint>&);
void getSegmentation(Ptr<cv::ximgproc::segmentation::GraphSegmentation>, Mat, Mat, Mat&, Mat&);
void getGroups(Mat, vector<vector<Point> >&, vector<vector<Point> >&, vector<Point>&, vector<KeyPoint>, vector<KeyPoint>);

/* Helper function for object avoidance. */
void getMovement(Mat, Mat, Mat, vector<vector<Point> >, vector<vector<Point> >, double&, double&, double&, double&);

/* Pthread function for object avoidance. */
/* This Pthread function recieves the x, y, z, and rotation parameters of UAV
 * and moves the UAV accordingly, for 5 iterations.
 * @param threadarg struct used to pass x, y, z, and rotation  parameters by value
*/
void *move(void *threadarg)
{
   thread_data *my_data;

	my_data = (thread_data *) threadarg;

	if (my_data->vyt != 0 || my_data->vzt != 0)
	{
		cout << "Stop-Turn..." << endl;
		for(int i = 0; i < 10; i++)
			ardrone.move3D(0, my_data->vyt, 0, 0);
	}
	else
		for(int i = 0; i < 10; i++)
			ardrone.move3D(0.2, 0, 0, 0);

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

		cv::Mat frame, currFrame, origFrame, prevFrame, h_currDescriptors, h_prevDescriptors;
		std::vector<cv::KeyPoint> currKeypoints, prevKeypoints;

		GpuMat d_frame, prevDescriptors, currDescriptors, d_keypoints;
		GpuMat d_threshold;

		/* Segmentation variable */
		Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg = cv::ximgproc::segmentation::createGraphSegmentation(0.5, 500, 50);

		SURF_CUDA detector(800);
		Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(detector.defaultNorm());

		vector<cv::DMatch> matches, good_matches;

		origFrame = ardrone.getImage();
		cv::resize(origFrame, prevFrame, cv::Size(200,200));

		getFeatures(detector, prevFrame, h_prevDescriptors, prevKeypoints);

		for (;;)
		{

			std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]\r" << std::flush;
			origFrame = ardrone.getImage();
			cv::resize(origFrame, currFrame, cv::Size(200,200));

			cv::imshow("Original", currFrame);

			clock_gettime(CLOCK_MONOTONIC, &start);
			getFeatures(detector, currFrame, h_currDescriptors, currKeypoints);

			vector<DMatch> good_matches;
			getMatches(h_prevDescriptors, h_currDescriptors, prevKeypoints, currKeypoints, matcher, good_matches);
			currFrame.copyTo(frame);
			double vx = 1.0, vy = 0.0, vz = 0.0, vr = 0.0;
			try
			{
				if (good_matches.size() > 2)
				{

					Mat prev_output_image, curr_output_image;
					std::vector<cv::KeyPoint> prev_expanding, curr_expanding;
					getTemplate(prevFrame, currFrame, prevKeypoints, currKeypoints, good_matches, prev_expanding, curr_expanding);

					getSegmentation(seg, prevFrame, currFrame, prev_output_image, curr_output_image);
					imshow("curr", curr_output_image);

					vector<vector<Point> > prev_groups;
					vector<vector<Point> > curr_groups;
					vector<Point> midpoints;
					getGroups(curr_output_image, prev_groups, curr_groups, midpoints, prev_expanding, curr_expanding);

					
					getMovement(frame, prev_output_image, curr_output_image, prev_groups, curr_groups, vx, vy, vz, vr);

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

				}

			}
			catch (const cv::Exception& ex)
			{

				continue;

			}

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


			/* Calculate individual FPS */
			storage = avgfps();

			cv::imshow("Result", frame);
			char key = cvWaitKey(1);

			if (key == 27) // Esc key
				break;

			currFrame.copyTo(prevFrame);
			prevKeypoints.clear();
			for (unsigned int i = 0; i < currKeypoints.size(); i++)
				prevKeypoints.push_back(currKeypoints[i]);
//			h_prevKeypoints = h_currKeypoints;
			currKeypoints.clear();
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
					ardrone.move3D(0, 0, -0.05, 0);
					sleep(3);
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
		double ratio = currKeys[curr].size / (double)prevKeys[prev].size;
		if (matches[i].distance < 3*min_dist  && ratio > 1 && ratio < 2 )
			good_matches.push_back(matches[i]);

	}

//	cout << "Matches: " << good_matches.size() << endl;

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

/**
* This function detects expanding features using template matching and stores result in expanding
* @param prevFrame The previous frame
* @param currFrame The current frame
* @param h_prevKeypoints The keypoints of the previous frame
* @param h_currKeypoints The keypoints of the current frame
* @param good_matches The container for the filtered out matches
* @param expanding Resulting keypoints which are a threat
*/
void getTemplate(Mat prevFrame, Mat currFrame, vector<KeyPoint> h_prevKeypoints, vector<KeyPoint> h_currKeypoints, vector<DMatch>& good_matches, vector<KeyPoint>& prev_expanding, vector<KeyPoint>& curr_expanding)
{

	Mat result;

	/* Used for convenient reference of previous and current keypoints */
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
	
	/* Template Matching */
	for(unsigned int i = 0; i < good_matches.size(); i++)
	{
		/* Draw template around previous feature */
//		double t_size1 = h_prevKeypoints[good_matches[i].queryIdx].size * (8.0 / 3.0);
		cv::Rect section1(prev_quick[i].Pt.x - 15, prev_quick[i].Pt.y - 15, 30, 30); 

		bool is_inside = (section1 & cv::Rect(0, 0, prevFrame.cols, prevFrame.rows)) == section1;
		if(!is_inside)
			continue;
		Mat PrevTemp = prevFrame(section1);

		double TMmin = 2, Scalemin = 2;
		for(double scale = 1; scale < 2; scale += 0.1)
		{

			/* Scale Previous feature */			
			Mat new_PrevTemp;
			cv::resize(PrevTemp, new_PrevTemp, cvSize(0, 0), scale, scale);

			/* Draw template around current feature */
//			double t_size2 = h_currKeypoints[good_matches[i].trainIdx].size * (8.0 / 3.0) * scale;

			cv::Rect section2(curr_quick[i].Pt.x - 15 * scale , curr_quick[i].Pt.y - 15 * scale,
									  30 * scale, 30 * scale);

			bool is_inside2 = (section2 & cv::Rect(0, 0, currFrame.cols, currFrame.rows)) == section2;
			if(!is_inside2)
				continue;
			Mat CurrTemp = currFrame(section2);

			matchTemplate(new_PrevTemp, CurrTemp, result, CV_TM_SQDIFF_NORMED);		

	 		double minVal, maxVal, TMscale; 
			Point minLoc,  maxLoc, matchLoc;
	 		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
			TMscale = minVal/(scale*scale);


			if(TMscale < TMmin)
			{
				TMmin = TMscale;
				Scalemin = scale;
			}
	
//			cout<< "Feature: " << i << " Scale: " << scale << " Min: " << minVal << endl;
		}
		
		if(Scalemin > 1.2 && TMmin < 0.8)
		{
			prev_expanding.push_back( h_prevKeypoints[good_matches[i].queryIdx]);
			curr_expanding.push_back( h_currKeypoints[good_matches[i].trainIdx]);
		}
	}

	Mat sample;
	currFrame.copyTo(sample);
	//for (unsigned int i = 0; i < curr_expanding.size(); i++)
	//	circle(sample, curr_expanding[i].pt, 6, Scalar(200, 25, 20), -1);
	drawKeypoints(sample, h_currKeypoints, sample, Scalar(25, 255, 2) );
	drawKeypoints(sample, curr_expanding, sample, Scalar(2, 25, 200) );
	imshow("Expanding", sample);

}

void getGroups(Mat curr_output_image, vector<vector<Point> >& prev_groups, vector<vector<Point> >& curr_groups, vector<Point>& midpoints, vector<KeyPoint> prev_expanding, vector<KeyPoint> curr_expanding)
{

	vector<node> prev_quick;
	vector<node> curr_quick;

    /* Grab and store the good matches' keypoints from both frames */
	for (unsigned int i = 0; i < curr_expanding.size(); i++)
	{

		node prev_node = {false, prev_expanding[i].pt, -1};
		node curr_node = {false, curr_expanding[i].pt, -1};
		prev_quick.push_back(prev_node);
		curr_quick.push_back(curr_node);

	}

	/*-------------- Clustering Section ---------------*/

   /* Set up containers for the previous and current clusters */
	//std::vector<cv::Point> prev_cluster;
	//std::vector<cv::Point > curr_cluster;


	/* Cluster points into groups by color */
	for(unsigned int i = 0; i < curr_quick.size(); i++)
	{

		int FLAG2 = 0;

		for(unsigned int j = 0; j < curr_groups.size(); j++)
		{
			if(curr_output_image.at<Vec3b>(curr_expanding[i].pt) == 
				curr_output_image.at<Vec3b>(curr_groups[j][0]))
			{
				curr_groups[j].push_back(curr_expanding[i].pt);
				prev_groups[j].push_back(prev_expanding[i].pt);
				FLAG2 = 1;
			}
		}
		
		if( curr_groups.size() == 0 || FLAG2 == 0);
		{
			std::vector<cv::Point> prev_cluster;
			std::vector<cv::Point > curr_cluster;
			curr_cluster.push_back(curr_expanding[i].pt);
			prev_cluster.push_back(prev_expanding[i].pt);
			prev_groups.push_back(prev_cluster);
			curr_groups.push_back(curr_cluster);
		}
			

	}

	/* Compute midpoint for the cluster 
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

		/* Add the cluster and midpoint to their respective containers 
		if (currGroups[p].size() > 3)
		{
			curr_midpoints.push_back(c_mid);
			prev_midpoints.push_back(p_mid);
		}

	}*/
}

void getMovement(Mat frame, Mat prev_output_image, Mat curr_output_image, vector<vector<Point> > prev_groups, vector<vector<Point> > curr_groups, double& vx, double& vy, double& vz, double& vr)
{

	FLAG = 0;

	for (unsigned int i = 0; i < curr_groups.size(); i++)
	{

		int prev_fill = 0, curr_fill = 0;
//		Rect boundRect;
		clock_gettime(CLOCK_MONOTONIC, &finish);
		elapsed = (finish.tv_sec - start.tv_sec);
		elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

//		prev_fill = floodFill(prev_output_image, prev_groups[i][0], cv::Scalar(250, 250, 250), &boundRect);
//		curr_fill = floodFill(curr_output_image, curr_groups[i][0], cv::Scalar(250, 250, 250), &boundRect);
//		cv::circle(frame, curr_groups[i][0], 10, cv::Scalar(250, 50, 0), 3);

		// Calculate area of expansion of detected objects
//		double ratio = (double)curr_fill / prev_fill;


		//cout << "Ratio: " << ratio << endl;

		// Used to approximate countours to polygons + get bounding rects
		std::vector<std::vector<cv::Point> > hull(2);
		cv::convexHull(prev_groups[i], hull[0], false);
		cv::convexHull(curr_groups[i], hull[1], false);
/*		cv::drawContours(frame, hull, 0, cv::Scalar(255, 0, 0), 1, 8,
							std::vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(frame, hull, 1, cv::Scalar(20, 255, 0), 1, 8,
							std::vector<cv::Vec4i>(), 0, cv::Point());
*/
		cv::Rect boundRect;

		double ratio = cv::contourArea(hull[1]) / cv::contourArea(hull[0]);
		cout << "Ratio: " << ratio << "Hull 1: " << cv::contourArea(hull[1])
				<< "Hull 0: " << cv::contourArea(hull[0]) << endl;

		if (ratio > 1 && ratio < 2 )
		{
			floodFill(curr_output_image, curr_groups[i][0], cv::Scalar(250, 250, 250), &boundRect);


			FLAG = 1;

			// Used to approximate countours to polygons + get bounding rect			
			cv::rectangle(frame, boundRect.tl(), boundRect.br(),
							 cv::Scalar(200,200,0), 2, 8, 0);

			// Check which quadrant(s) rectangle is in
			cv::Point top_left(boundRect.tl().x, boundRect.tl().y);
			cv::Point top_right(boundRect.tl().x + boundRect.width, boundRect.tl().y);
			cv::Point bot_left(boundRect.tl().x, boundRect.tl().y + boundRect.height);
			cv::Point bot_right(boundRect.tl().x + boundRect.width, boundRect.tl().y + boundRect.height);

			vy += min(1.0, 0.1 * (top_left.x - frame.cols / 2.0));
			vy += max(-1.0, 0.1 * (top_right.x - frame.cols / 2.0));

//       vz -= min(1.0, 0.01 * (top_left.y - frame.rows / 2.0));
//       vz -= max(-1.0, 0.01 * (bot_left.y - frame.rows / 2.0));
			cout << "Left/Right: " << vy << " Up/Down: " << vz << endl;
		}

	}


}

/* --------------------------- SCRAP CODE ------------------------- */


	/* Draw Template */
/*	Mat sample;
	currFrame.copyTo(sample);
	rectangle( sample, matchLoc, Point( matchLoc.x + CurrTemp.cols , matchLoc.y + CurrTemp.rows ), Scalar::all(0), 2, 8, 0 );
	rectangle( result, matchLoc, Point( matchLoc.x + CurrTemp.cols , matchLoc.y + CurrTemp.rows ), Scalar::all(0), 2, 8, 0 );
	imshow( "image_window", sample );
	imshow( "result_window", result );
	imshow("PrevTemp", new_PrevTemp);
	imshow("CurrTemp", CurrTemp);

	imshow("CurrTemp", CurrTemp);

	cout<< "Feature: " << i << " Scale: " << scale << " Min: " << minVal << endl;
*/

	/* Size resulting matrix for matching */
/*
	int result_cols = CurrTemp.cols - new_PrevTemp.cols + 1;
	int result_rows = CurrTemp.rows - new_PrevTemp.rows + 1;
	cv::cvtColor(new_PrevTemp, new_PrevTemp, CV_8UC1);
	cv::cvtColor(CurrTemp, CurrTemp, CV_8UC1);
	result.create( result_rows, result_cols, CV_32FC1 );
*/	 
	

	/* Alternate template match */
/*	cv::Rect section2(curr_quick[i].Pt.x - scale2/2 , curr_quick[i].Pt.y - scale2/2 ,
							scale2, scale2);
	cv::Rect roi2 = section2 & cv::Rect(0, 0, currFrame.cols, currFrame.rows);
	Mat CurrTemp = currFrame(roi2);
*/


	/* Alternate template match */
/*	cv::Rect section1(prev_quick[i].Pt.x - scale1/2 , prev_quick[i].Pt.y - scale1/2 ,
								scale1, scale1);
	cv::Rect roi1 = section1 & cv::Rect(0, 0, prevFrame.cols, prevFrame.rows);
	Mat PrevTemp = prevFrame(roi1);
	cv::Rect section2(curr_quick[i].Pt.x - 5, curr_quick[i].Pt.y - 5, 10 * scale, 10 * scale); */ 

/* -------Begining of Line Section Before Segmentation-------*/
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
