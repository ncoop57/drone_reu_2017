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
#include <pthread.h>
//#include <omp.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;

#define NUM_THREADS 1

ARDrone ardrone;

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

void getFeatures(Mat, Mat&, vector<KeyPoint>&);

// Pthread function to issue avoidance command
void *move(void *threadarg)
{
   thread_data *my_data;

	my_data = (thread_data *) threadarg;
	double vx = my_data->vxt;
	double vy = my_data->vyt;
	double vz = my_data->vzt;
	double vr = my_data->vrt;

	for(int i = 0; i < 5; i++)
		ardrone.move3D(0.0, 0.0, 0.0, 1.0);
	pthread_exit(NULL);
}

int main(int argc, char* argv[])
{

	try
	{


//		ARDrone ardrone;

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

			std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]\r" << std::flush;
			currFrame = ardrone.getImage();
			getFeatures(currFrame, h_currDescriptors, h_currKeypoints);
			/*d_frame.upload(currFrame);
			cv::cuda::cvtColor(d_frame, d_greyFrame, CV_RGB2GRAY);

			detector(d_greyFrame, GpuMat(), d_keypoints, d_descriptors);
			detector.downloadKeypoints(d_keypoints, h_currKeypoints);
			d_descriptors.download(h_currDescriptors);*/

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

			try
			{
				if (good_matches.size() > 2)
				{				
					std::vector<cv::Point> prevPoints;
					std::vector<cv::Point> currPoints;

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

//					cout << "Initialized Nodes" << endl;
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

//							cout << " Q Size: " << queue.size() << endl;

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


						if (curr_cluster.size() > 2)
						{

							prev_group_pts.push_back(prev_cluster);
							curr_group_pts.push_back(curr_cluster);
					//		cout << "Group Created: " << curr_group_pts.size() << endl;

						}

					}
						
	//				cout << "Group Created: " << curr_group_pts.size() << endl;
/*
					for(unsigned int i = 0; i < curr_group_pts.size(); i++)
					{
						cout << "Group # " << i << " , Count : " << curr_group_pts[i].size() << endl;

					}
					
					cout << " --------------------------------------- " << endl << endl;
				
*/					// --------------End of Clustering----------------- 



					double vx = 1.0, vy = 0.0, vz = 0.0, vr = 0.0;
					std::vector<std::vector<cv::Point> > prev_hull(prev_group_pts.size());
					std::vector<std::vector<cv::Point> > curr_hull(curr_group_pts.size());
					for (unsigned int i = 0; i < curr_group_pts.size(); i++)
					{

						cv::convexHull(prev_group_pts[i], prev_hull[i], false);
						cv::convexHull(curr_group_pts[i], curr_hull[i], false);
//						cv::drawContours(frame, hull, i, cv::Scalar(0, 0, 255), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
						double prev_area = cv::contourArea(prev_hull[i]);
						double curr_area = cv::contourArea(curr_hull[i]);
						double ratio = curr_area / prev_area;
//				              	std::cout << "Prev Area: " << prev_area << " Curr Area: " << curr_area << " Ratio: " << ratio << std::endl;
						if (ratio > 1)
						{
							cv::drawContours(frame, curr_hull, i, cv::Scalar(255, 0, 0), 1, 8,
									std::vector<cv::Vec4i>(), 0, cv::Point());

							// Used to approximate countours to polygons + get bounding rects
							std::vector<cv::Point> contours_poly;
							cv::Rect boundRect;

							boundRect = cv::boundingRect(cv::Mat(curr_hull[i]));
							cv::rectangle(frame, boundRect.tl(), boundRect.br(),
											 cv::Scalar(200,200,0), 2, 8, 0);

							// Check which quadrant(s) rectangle is in
							cv::Point top_left(boundRect.tl().x, boundRect.tl().y);
							cv::Point top_right(boundRect.tl().x + boundRect.width, boundRect.tl().y);
							cv::Point bot_left(boundRect.tl().x, boundRect.tl().y + boundRect.height);
							cv::Point bot_right(boundRect.tl().x + boundRect.width, boundRect.tl().y + boundRect.height);

							vy += min(5.0, 0.1 * (top_left.x - frame.cols / 2.0));
							vy += max(-5.0, 0.1 * (top_right.x - frame.cols / 2.0));

	//                                              vz -= min(1.0, 0.01 * (top_left.y - frame.rows / 2.0));
	//                                              vz -= max(-1.0, 0.01 * (bot_left.y - frame.rows / 2.0));
							cout << "Left/Right: " << vy << " Up/Down: " << vz << endl;

						}

					}


					
/*					vx = 1.0;
					vy = 0.0;
					vz = 0.0;
					vr = 0.0;	
					for(int q = 0; q < 5; q++)
						ardrone.move3D(vx, vy, vz, vr);
*/				
					
			//		#pragma omp parallel for
			/* ------------------PTHREAD SECTION ----------------------*/

					pthread_t threads[NUM_THREADS];
					int rc;
					long t;

					for(t = 0; t < NUM_THREADS; t++)
					{

						thread_data_array[t].vxt = 0.0;
						thread_data_array[t].vyt = 0.0;
						thread_data_array[t].vzt = 0.0;
						thread_data_array[t].vrt = 0.0;

						rc = 	pthread_create(&threads[t], NULL, move, 
								(void *) &thread_data_array[t]); 
						if(rc){
							cout << "Error creating thread" << endl;
							exit(-1);
						}
					} 




/*
					for (int i = 0; i < 10; i++)
					{

						char key = cv::waitKey(1);

						if (key == ' ') {
							if (ardrone.onGround()) ardrone.takeoff();
							else                    ardrone.landing();
						}

		
					ardrone.move3D(vx, vy, vz, vr);

					}

*/			/*------------------END OF PTHREAD-------------------------*/
					curr_group_pts.clear();
					prev_group_pts.clear();

				}

			}
			catch (const cv::Exception& ex)
			{

				continue;

			}


			cv::imshow("Result", frame);
			char key = cv::waitKey(1);

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
					sleep(15);
					cout << "End" << endl;
				}
				else ardrone.landing();			
			}
        		static int mode = 0;
        		if (key == 'c') ardrone.setCamera(++mode % 4);


		}


		ardrone.close();

	}
	catch (const cv::Exception& ex)
	{

		std::cout << "Error: " << ex.what() << std::endl;

	}
	
//	pthread_exit(NULL);
	return 0;
}

void getFeatures(Mat frame, Mat& descriptors, vector<KeyPoint>& keys)
{

	SURF_CUDA detector(400);
	GpuMat d_frame, d_keypoints, d_descriptors;

	d_frame.upload(frame);
	cuda::cvtColor(d_frame, d_frame, CV_RGB2GRAY);

	detector(d_frame, GpuMat(), d_keypoints, d_descriptors);
	detector.downloadKeypoints(d_keypoints, keys);
	d_descriptors.download(descriptors);

}
