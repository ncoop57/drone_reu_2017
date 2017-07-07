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
//#include <omp.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;

#define NUM_THREADS 1

ARDrone ardrone;
int FLAG = 0;

/* Used for segmentation */
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
		ardrone.move3D(0.0, vy, vz, vr);
	pthread_exit(NULL);
}


int main(int argc, char* argv[])
{

	try
	{


//		ARDrone ardrone;

		if (!ardrone.open())
			return -1;
		
		cv::Mat frame, currFrame, origFrame, prevFrame, h_currDescriptors, h_prevDescriptors, image_gray;
		std::vector<cv::KeyPoint> h_currKeypoints, h_prevKeypoints;

		GpuMat d_frame, d_fgFrame, d_greyFrame, d_descriptors, d_keypoints;
		GpuMat d_threshold;

		/* Segmentation variable */
		Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg = cv::ximgproc::segmentation::createGraphSegmentation(0.8, 300, 30);


		SURF_CUDA detector(800);
		cv::BFMatcher matcher;
		//Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(detector.defaultNorm());

		vector<cv::DMatch> matches, good_matches;

		origFrame = ardrone.getImage();
		cv::resize(origFrame, prevFrame, cv::Size(200,200));

		d_frame.upload(prevFrame);
		cv::cuda::cvtColor(d_frame, d_greyFrame, CV_RGB2GRAY);

		detector(d_greyFrame, GpuMat(), d_keypoints, d_descriptors);
		

		detector.downloadKeypoints(d_keypoints, h_prevKeypoints);
		d_descriptors.download(h_prevDescriptors);

		for (;;)
		{

			std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]\r" << std::flush;
			origFrame = ardrone.getImage();
			cv::resize(origFrame, currFrame, cv::Size(200,200));


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
		
			for (int i = 0; i < h_prevDescriptors.rows; i++)
			{

				int prev = matches[i].queryIdx;
				int curr = matches[i].trainIdx;

				double ratio = h_currKeypoints[curr].size / h_prevKeypoints[prev].size;
				if (matches[i].distance < 4*min_dist && ratio > 1.0)
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

					std::vector<cv::Point> prev_midpoints;
					std::vector<cv::Point> curr_midpoints;
					

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
						
						// Compute midpoints
						double cx = 0, cy = 0, px = 0, py = 0;
						for(unsigned int p = 0; p < curr_cluster.size(); p++)
						{
							cx += curr_cluster[p].x;
							cy += curr_cluster[p].y;

							px += prev_cluster[p].x;
							py += prev_cluster[p].y;

						}
						
						cx /= curr_cluster.size();
						cy /= curr_cluster.size();
						px /= curr_cluster.size();
						py /= curr_cluster.size();

						Point c_mid(cx, cy);
						Point p_mid(px, py);

						if (curr_cluster.size() > 2)
						{
							curr_midpoints.push_back(c_mid);
							prev_midpoints.push_back(p_mid);

							prev_group_pts.push_back(prev_cluster);
							curr_group_pts.push_back(curr_cluster);
					//		cout << "Group Created: " << curr_group_pts.size() << endl;

						}

					}
						
	//				cout << "Group Created: " << curr_group_pts.size() << endl;

					// --------------End of Clustering----------------- 

					// -------------Begining of Segmentation ----------


		Mat prev_input, prev_output, curr_input, curr_output, prev_output_image, curr_output_image;

		seg->processImage(prevFrame, prev_output);


		double mins, maxs;
		minMaxLoc(prev_output, &mins, &maxs);

		int nb_segs = (int)maxs + 1;

		std::cout << nb_segs << " segments" << std::endl;

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


		seg->processImage(currFrame, curr_output);

		minMaxLoc(curr_output, &mins, &maxs);

		nb_segs = (int)maxs + 1;

		std::cout << nb_segs << " segments" << std::endl;

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



		imshow("curr", curr_output_image);


					// ------------End of segmentation ---------------






					double vx = 1.0, vy = 0.0, vz = 0.0, vr = 0.0;
					std::vector<std::vector<cv::Point> > prev_hull(prev_group_pts.size());
					std::vector<std::vector<cv::Point> > curr_hull(curr_group_pts.size());
					for (unsigned int i = 0; i < curr_group_pts.size(); i++)
					{

						int prev_fill = 0, curr_fill = 0;
						Rect boundRect;
						

						prev_fill = floodFill(prev_output_image, prev_midpoints[i], cv::Scalar(250, 250, 250));
						curr_fill = floodFill(curr_output_image, curr_midpoints[i], cv::Scalar(250, 250, 250), &boundRect);
						cv::circle(frame, curr_midpoints[i], 10, cv::Scalar(250, 50, 0), 3);

						double ratio = (double)curr_fill / prev_fill;
				      std::cout << "Prev Area: " << prev_fill << " Curr Area: " << curr_fill << " Ratio: " << ratio << std::endl;
						if (ratio > 1.2 && ratio < 2)
						{

							// Used to approximate countours to polygons + get bounding rects

							cv::rectangle(frame, boundRect.tl(), boundRect.br(),
											 cv::Scalar(200,200,0), 2, 8, 0);

							// Check which quadrant(s) rectangle is in
							cv::Point top_left(boundRect.tl().x, boundRect.tl().y);
							cv::Point top_right(boundRect.tl().x + boundRect.width, boundRect.tl().y);
							cv::Point bot_left(boundRect.tl().x, boundRect.tl().y + boundRect.height);
							cv::Point bot_right(boundRect.tl().x + boundRect.width, boundRect.tl().y + boundRect.height);

							vy += min(0.25, 0.01 * (top_left.x - frame.cols / 2.0));
							vy += max(-0.25, 0.01 * (top_right.x - frame.cols / 2.0));

	//                                              vz -= min(1.0, 0.01 * (top_left.y - frame.rows / 2.0));
	//                                              vz -= max(-1.0, 0.01 * (bot_left.y - frame.rows / 2.0));
							cout << "Left/Right: " << vy << " Up/Down: " << vz << endl;

						}

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

		        // Move
/*		        double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
        		if (key == 'w' || key == CV_VK_UP)    vx =  0.5;
	        	if (key == 's' || key == CV_VK_DOWN)  vx = -0.5;
	       	if (key == 'q' || key == CV_VK_LEFT)  vr =  0.5;
		 		if (key == 'e' || key == CV_VK_RIGHT) vr = -0.5;
       		if (key == 'a') vy =  1.0;
       		if (key == 'd') vy = -1.0;
        		if (key == 'u') vz =  1.0;
        		if (key == 'j') vz = -1.0;
        		ardrone.move3D(vx, vy, vz, vr);
*/
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
	
//	pthread_exit(NULL);
	return 0;
}


