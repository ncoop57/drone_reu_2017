#include <iostream>
#include <stdio.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/features2d.hpp>
#include "ardrone/ardrone.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

typedef struct{

	double roll;
	double pitch;
	double yaw;
	double vert_speed;
}Navigation;

typedef struct{
	bool A;
	bool B;
	bool C;
	bool D;
}obs_location;

typedef struct{
	int x_pos;
	int y_pos;
	int z_pos;
}location;

// Constant floats used within Obstacle_Avoidance
const float Kph = 0.5;
const float Kp0 = 0.5;

Navigation Obstacle_Avoidance(obs_location img_loc, location uav_loc);

int main(int argc, char* argv[])
{

	try
	{

//		cv::VideoCapture cap(0);

		ARDrone ardrone;

		if (!ardrone.open())
			return -1;
		
		cv::Mat frame, currFrame, prevFrame, h_currDescriptors, h_prevDescriptors, image_gray;
		std::vector<cv::KeyPoint> h_currKeypoints, h_prevKeypoints;

		GpuMat d_frame, d_fgFrame, d_greyFrame, d_descriptors, d_keypoints;
		GpuMat d_threshold;

		SURF_CUDA detector(1000);
		cv::BFMatcher matcher;
		//Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(detector.defaultNorm());

		vector<cv::DMatch> matches, good_matches;

		prevFrame = ardrone.getImage();

		d_frame.upload(prevFrame);
		cv::cuda::cvtColor(d_frame, d_greyFrame, CV_RGB2GRAY);

		detector(d_greyFrame, GpuMat(), d_keypoints, d_descriptors);
		

		detector.downloadKeypoints(d_keypoints, h_prevKeypoints);
		d_descriptors.download(h_prevDescriptors);

		for (;;)
		{


			currFrame = ardrone.getImage();

			cv::cvtColor(currFrame, image_gray, CV_BGR2GRAY);
			cv::blur(image_gray, image_gray, Size(3,3));
			cv::Canny(image_gray, image_gray, 10, 30, 3);			
			
			d_frame.upload(currFrame);

		
			cv::cuda::cvtColor(d_frame, d_greyFrame, CV_RGB2GRAY);

			detector(d_greyFrame, GpuMat(), d_keypoints, d_descriptors);
			detector.downloadKeypoints(d_keypoints, h_currKeypoints);
			d_descriptors.download(h_currDescriptors);

			h_prevDescriptors.convertTo(h_prevDescriptors, CV_32F);
			h_currDescriptors.convertTo(h_currDescriptors, CV_32F);


			try
			{
			
				matcher.match(h_prevDescriptors, h_currDescriptors, matches);

			}
			catch (const cv::Exception& ex)
			{

				;

			}

			currFrame.copyTo(frame);

			double max_dist = 0;
			double min_dist = 100;


			for (int i = 0; i < h_prevDescriptors.rows; i++)
			{

				double dist = matches[i].distance;
				if (dist < min_dist)
					min_dist = dist;

				if (dist > max_dist)
					max_dist = dist;

			}

			std::vector<cv::DMatch> good_matches;
			for (int i = 0; i < h_prevDescriptors.rows; i++)
			{

				int prev = matches[i].queryIdx;
				int curr = matches[i].trainIdx;

				double ratio = h_currKeypoints[curr].size / h_prevKeypoints[prev].size;
				if (matches[i].distance < 4*min_dist && ratio > 1.5)
				{

					good_matches.push_back(matches[i]);

				}

			}


			try
			{
				if (good_matches.size() > 2)
				{				
					std::vector<cv::Point> prevPoints;
					std::vector<cv::Point> currPoints;
					std::vector<std::vector<cv::Point> > hull(2);

					for (int i = 0; i < good_matches.size(); i++)
					{

						prevPoints.push_back(h_currKeypoints[good_matches[i].queryIdx].pt);
						currPoints.push_back(h_currKeypoints[good_matches[i].trainIdx].pt);

					}

					cv::convexHull(prevPoints, hull[0], false);
					cv::convexHull(currPoints, hull[1], false);
				
					double area = cv::contourArea(hull[1]) / cv::contourArea(hull[0]);
					std::cout << "Area: " << area << std::endl;
					if (area > 0.8)
					{ 
						cv::drawContours(frame, hull, 0, cv::Scalar(255, 0, 0), 1, 8,
								std::vector<cv::Vec4i>(), 0, cv::Point());

						// Used to approximate countours to polygons + get bounding rects
						std::vector<cv::Point> contours_poly;
						cv::Rect boundRect;
	
						cv::approxPolyDP(cv::Mat(hull[1]), contours_poly, 3, true);
						boundRect = cv::boundingRect(cv::Mat(contours_poly));
						
						cv::rectangle(frame, boundRect.tl(), boundRect.br(),
										 cv::Scalar(200,200,0), 2, 8, 0);

						// Check which quadrant(s) rectangle is in
						obs_location rect_loc = {false, false, false, false};

						if(boundRect.br().x  >= frame.cols/2.0 && boundRect.br().y >= frame.rows/2.0)
							rect_loc.A = true;
						if(boundRect.br().x > frame.cols/2.0 && boundRect.tl().y < frame.cols/2.0)
							rect_loc.B = true;
						if(boundRect.tl().x <= frame.cols/2.0 && boundRect.br().y >= frame.rows/2.0)
							rect_loc.C = true;
						if(boundRect.tl().x < frame.cols/2.0 && boundRect.tl().y < frame.cols/2.0)
							rect_loc.D = true;

						location uav_loc = {0,0,0};
						Navigation command  = Obstacle_Avoidance(rect_loc, uav_loc);
					}
					
					if(area > 2)
						std::cout << "HALT" << std::endl;
				}

			}
			catch (const cv::Exception& ex)
			{

				continue;

			}


			cv::imshow("Result", frame);
			char key = cv::waitKey(60);

			if (key == 27) // Esc key
				break;

			currFrame.copyTo(prevFrame);
			h_prevKeypoints = h_currKeypoints;
			h_currDescriptors.copyTo(h_prevDescriptors);

			// Take off / Landing 
        		if (key == ' ') {
        	    		if (ardrone.onGround()) ardrone.takeoff();
           	 		else                    ardrone.landing();
        		}

		        // Move
		        double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
        		if (key == 'w' || key == CV_VK_UP)    vx =  1.0;
	        	if (key == 's' || key == CV_VK_DOWN)  vx = -1.0;
        		if (key == 'q' || key == CV_VK_LEFT)  vr =  1.0;
			if (key == 'e' || key == CV_VK_RIGHT) vr = -1.0;
       			if (key == 'a') vy =  1.0;
       			if (key == 'd') vy = -1.0;
        		if (key == 'u') vz =  1.0;
        		if (key == 'j') vz = -1.0;
        		ardrone.move3D(vx, vy, vz, vr);

        		// Change camera
        		static int mode = 0;
        		if (key == 'c') ardrone.setCamera(++mode % 4);


		}


		ardrone.close();

	}
	catch (const cv::Exception& ex)
	{

		std::cout << "Error: " << ex.what() << std::endl;

	}

	return 0;

}

Navigation Obstacle_Avoidance(obs_location img_loc, location uav_loc){

	Navigation command ={0,0,0,0};

	// Series of if conditions: starting from single quadrant and 
	// ending at all quadrants being occupied
	if(img_loc.A == true)
		std::cout << "A";
	if(img_loc.B == true)
		std::cout << "B";
	if(img_loc.C == true)
		std::cout << "C";
	if(img_loc.D == true)
		std::cout << "D";

	std::cout << std::endl;
	
	if((img_loc.D == false || img_loc.C == false) && 
			(img_loc.A = true || img_loc.B == true || (img_loc.A == true && img_loc.B == true)))
		std::cout << "Move: LEFT" << std::endl;	

	else if((img_loc.A == false || img_loc.B == false) &&
			 (img_loc.C = true || img_loc.D == true || (img_loc.C == true && img_loc.D == true)))
		std::cout << "Move: RIGHT" << std::endl;

	if((img_loc.B == false || img_loc.D == false) && 
			img_loc.A == true || img_loc.C == true || (img_loc.A == true && img_loc.C == true))
		std::cout << "Move: UP" << std::endl;

	else if((img_loc.A == false || img_loc.C == false) &&
			img_loc.B == true || img_loc.D == true || (img_loc.B == true && img_loc.D == true))
		std::cout << "Move: DOWN" << std::endl;

/*	else if(img_loc.A == true && img_loc.B == true && img_loc.C == true)
		std::cout << "Move: UPPER-LEFT" << std::endl;

	else if(img_loc.A = true && img_loc.B == true && img_loc.D == true)
		std::cout << "Move: LOWER-LEFT" << std::endl;

	else if(img_loc.A = true && img_loc.C == true && img_loc.D == true)
		std::cout << "Move: UPPER-RIGHT" << std::endl;

	else if(img_loc.B = true && img_loc.C == true && img_loc.D == true)
		std::cout << "Move: LOWER-RIGHT" << std::endl;
*/
	 if(img_loc.A = true && img_loc.B == true && img_loc.C == true && img_loc.D == true)
		//	Navigation next_Way_Point = Receive next waypoint	
		if(0 >= uav_loc.y_pos)
			std::cout << "Move: LEFT -> ABCD" << std::endl;
		else
			std::cout << "Move: RIGHT -> ABCD" << std::endl;

	return command;
}
