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
#include "ardrone/ardrone.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

// Struct used in clustering section
typedef struct{
	bool is_grouped;
	cv::Point Pt;
	int ID;
}node;


int main(int argc, char* argv[])
{

	try
	{


		ARDrone ardrone;

		if (!ardrone.open())
			return -1;
		
		cv::Mat frame, currFrame, prevFrame, h_currDescriptors, h_prevDescriptors, image_gray;
		std::vector<cv::KeyPoint> h_currKeypoints, h_prevKeypoints;

		GpuMat d_frame, d_fgFrame, d_greyFrame, d_descriptors, d_keypoints;
		GpuMat d_threshold;

		SURF_CUDA detector(400);
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

			//std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]\r" << std::flush;
			currFrame = ardrone.getImage();

	//		cv::cvtColor(currFrame, image_gray, CV_BGR2GRAY);
	//		cv::blur(image_gray, image_gray, Size(3,3));
	//		cv::Canny(image_gray, image_gray, 10, 30, 3);			
			
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
		
//			cout << "KeyPts: " << h_currKeypoints.size() << endl;			

			for (int i = 0; i < h_prevDescriptors.rows; i++)
			{

				int prev = matches[i].queryIdx;
				int curr = matches[i].trainIdx;

				double ratio = h_currKeypoints[curr].size / h_prevKeypoints[prev].size;
				if (matches[i].distance < 4*min_dist && ratio > 1)
				{

					good_matches.push_back(matches[i]);

				}

			}

			cout << "GdMtchs: " << good_matches.size() << endl;
			
			try
			{
				if (good_matches.size() > 2)
				{				
					std::vector<cv::Point> prevPoints;
					std::vector<cv::Point> currPoints;
					//std::vector<std::vector<cv::Point> > hull(2);

					std::vector<node> prev_quick;  					
					std::vector<node> curr_quick;

					for (unsigned int i = 0; i < good_matches.size(); i++)
					{

						prevPoints.push_back(h_prevKeypoints[good_matches[i].queryIdx].pt);
						currPoints.push_back(h_currKeypoints[good_matches[i].trainIdx].pt);
						
						node prev_node = {false, h_prevKeypoints[good_matches[i].queryIdx].pt, -1};		
						node curr_node = {false, h_currKeypoints[good_matches[i].trainIdx].pt, -1};						
						prev_quick.push_back(prev_node);
						curr_quick.push_back(curr_node);
					}

					cout << "Initialized Nodes" << endl;
					//-------------- Clustering Section ---------------

					std::vector<std::vector<cv::Point > > prev_group_pts;
					std::vector<std::vector<cv::Point> > curr_group_pts;
					std::vector<node> queue;
					int threshold = 55;

					for(unsigned int i = 0; i < good_matches.size(); i++)
					{

					
						if(curr_quick[i].is_grouped)
							continue;

						std::vector<cv::Point> prev_cluster;
						std::vector<cv::Point > curr_cluster;	
						prev_cluster.push_back(prev_quick[i].Pt);
						curr_cluster.push_back(curr_quick[i].Pt);					
						curr_quick[i].is_grouped = true;						
						curr_quick[i].ID = curr_group_pts.size();


						queue.push_back(curr_quick[i]);

						while(!queue.empty())
						{

							cout << " Q Size: " << queue.size() << endl;

							node work_node = queue.back();
							queue.pop_back();

							for (unsigned int j = 0; j < good_matches.size(); j++)
							{
								if(work_node.Pt == curr_quick[j].Pt ||
								   curr_quick[j].is_grouped == true )
									continue;

								double dist = norm(work_node.Pt - curr_quick[j].Pt);
								if(dist < threshold)
								{
									curr_quick[j].is_grouped = true;
									curr_quick[j].ID = curr_group_pts.size();
									prev_cluster.push_back(prev_quick[j].Pt);
									curr_cluster.push_back(curr_quick[j].Pt);
									queue.push_back(curr_quick[j]);
								}

							}
			//				cout << "Added Node to cluster" << endl;


						}

						prev_group_pts.push_back(prev_cluster);
						curr_group_pts.push_back(curr_cluster);
						cout << "Group Created: " << curr_group_pts.size() << endl;

					}
						
					cout << "Group Created: " << curr_group_pts.size() << endl;

					for(unsigned int i = 0; i < curr_group_pts.size(); i++)
					{
						cout << "Group # " << i << " , Count : " << curr_group_pts[i].size() << endl;

					}
					
					cout << " --------------------------------------- " << endl << endl;
				
					// --------------End of Clustering----------------- 


					/*


						DRAW CLUSTERS


					*/					


					double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
					std::vector<std::vector<cv::Point> > prev_hull(prev_group_pts.size());
					std::vector<std::vector<cv::Point> > curr_hull(curr_group_pts.size());
					for (int i = 0; i < curr_group_pts.size(); i++)
					{

						cv::convexHull(prev_group_pts[i], prev_hull[i], false);
						cv::convexHull(curr_group_pts[i], curr_hull[i], false);
//						cv::drawContours(frame, hull, i, cv::Scalar(0, 0, 255), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
						double prev_area = cv::contourArea(prev_hull[i]);
						double curr_area = cv::contourArea(curr_hull[i]);
						double ratio = curr_area / prev_area;
				              	std::cout << "Prev Area: " << prev_area << " Curr Area: " << curr_area << " Ratio: " << ratio << std::endl;
						if (ratio > 1.3)
						{
							cv::drawContours(frame, curr_hull, i, cv::Scalar(255, 0, 0), 1, 8,
									std::vector<cv::Vec4i>(), 0, cv::Point());

							// Used to approximate countours to polygons + get bounding rects
							std::vector<cv::Point> contours_poly;
							cv::Rect boundRect;

						//      cv::approxPolyDP(cv::Mat(hull[1]), contours_poly, 3, true);
							boundRect = cv::boundingRect(cv::Mat(curr_hull[i]));

							cv::rectangle(frame, boundRect.tl(), boundRect.br(),
											 cv::Scalar(200,200,0), 2, 8, 0);

							// Check which quadrant(s) rectangle is in
							//obs_location rect_loc = {false, false, false, false};

							cv::Point top_left(boundRect.tl().x, boundRect.tl().y);
							cv::Point top_right(boundRect.tl().x + boundRect.width, boundRect.tl().y);
							cv::Point bot_left(boundRect.tl().x, boundRect.tl().y + boundRect.height);
							cv::Point bot_right(boundRect.tl().x + boundRect.width, boundRect.tl().y + boundRect.height);

							vr += min(1.0, 0.01 * (top_left.x - frame.cols / 2.0));
							vr += max(-1.0, 0.01 * (top_right.x - frame.cols / 2.0));

	//                                              vz -= min(1.0, 0.01 * (top_left.y - frame.rows / 2.0));
	//                                              vz -= max(-1.0, 0.01 * (bot_left.y - frame.rows / 2.0));
							cout << "Left/Right: " << vy << " Up/Down: " << vz << endl;

						}

					}


					for (int i = 0; i < 10; i++)
					{

						char key = cv::waitKey(60);

						if (key == ' ') {
							if (ardrone.onGround()) ardrone.takeoff();
							else                    ardrone.landing();
						}

						ardrone.move3D(vx, vy, vz, vr);

					}
					curr_group_pts.clear();
					prev_group_pts.clear();


					cout << "Draw the hulls" << endl;

					//cv::convexHull(prevPoints, hull[0], false);
					//cv::convexHull(currPoints, hull[1], false);
				
			/*		double area = cv::contourArea(hull[1]) / cv::contourArea(hull[0]);
			//		std::cout << "Area: " << area << std::endl;
					if (area > 1)
					{ 
						cv::drawContours(frame, hull, 0, cv::Scalar(255, 0, 0), 1, 8,
								std::vector<cv::Vec4i>(), 0, cv::Point());

						// Used to approximate countours to polygons + get bounding rects
						std::vector<cv::Point> contours_poly;
						cv::Rect boundRect;
	
					//	cv::approxPolyDP(cv::Mat(hull[1]), contours_poly, 3, true);
						boundRect = cv::boundingRect(cv::Mat(hull[1]));
						
						cv::rectangle(frame, boundRect.tl(), boundRect.br(),
										 cv::Scalar(200,200,0), 2, 8, 0);

						// Check which quadrant(s) rectangle is in
						//obs_location rect_loc = {false, false, false, false};

						cv::Point top_left(boundRect.tl().x, boundRect.tl().y);
						cv::Point top_right(boundRect.tl().x + boundRect.width, boundRect.tl().y);
						cv::Point bot_left(boundRect.tl().x, boundRect.tl().y + boundRect.height);
						cv::Point bot_right(boundRect.tl().x + boundRect.width, boundRect.tl().y + boundRect.height);

						double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
						vr += min(1.0, 0.01 * (top_left.x - frame.cols / 2.0));
						vr += max(-1.0, 0.01 * (top_right.x - frame.cols / 2.0));

//						vz -= min(1.0, 0.01 * (top_left.y - frame.rows / 2.0));
//						vz -= max(-1.0, 0.01 * (bot_left.y - frame.rows / 2.0));
						cout << "Left/Right: " << vy << " Up/Down: " << vz << endl;

						for (int i = 0; i < 10; i++)
						{

							char key = cv::waitKey(60);
	
							if (key == ' ') {
                                				if (ardrone.onGround()) ardrone.takeoff();
                               					else                    ardrone.landing();
                        				}

							ardrone.move3D(vx, vy, vz, vr);
						
						}
						//if(boundRect.br().x  >= frame.cols/2.0 && boundRect.br().y >= frame.rows/2.0)
						//	rect_loc.A = true;
						//if(boundRect.br().x > frame.cols/2.0 && boundRect.tl().y < frame.cols/2.0)
						//	rect_loc.B = true;
						//if(boundRect.tl().x <= frame.cols/2.0 && boundRect.br().y >= frame.rows/2.0)
						//	rect_loc.C = true;
						//if(boundRect.tl().x < frame.cols/2.0 && boundRect.tl().y < frame.cols/2.0)
						//	rect_loc.D = true;

						//location uav_loc = {0,0,0};
						//Navigation command  = Obstacle_Avoidance(rect_loc, uav_loc);
					}
					*/
//					if(area > 2)
//						std::cout << "HALT" << std::endl;
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
			h_prevKeypoints.clear();
			h_prevKeypoints = h_currKeypoints;
			h_currKeypoints.clear();
			h_currDescriptors.copyTo(h_prevDescriptors);

			// Take off / Landing 
        		if (key == ' ') {
        	    		if (ardrone.onGround()) ardrone.takeoff();
           	 		else                    ardrone.landing();
        		}

		        // Move
		        double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
        		if (key == 'w' || key == CV_VK_UP)    vx =  0.5;
	        	if (key == 's' || key == CV_VK_DOWN)  vx = -0.5;
        		if (key == 'q' || key == CV_VK_LEFT)  vr =  0.5;
			if (key == 'e' || key == CV_VK_RIGHT) vr = -0.5;
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


