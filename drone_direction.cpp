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
#include <stdlib.h>
#include <string>
#include <sstream>
#include <time.h>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::cuda;

#define NUM_THREADS 1

ARDrone ardrone;


// Struct used to pass arguments to Pthread
typedef struct{
	double vxt;
	double vyt;
	double vzt;
	double vrt;
	int itert;
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
	int iter = my_data-> itert;

	if(ardrone.onGround())
		pthread_exit(NULL);

	
	cout <<  " Values Passed: " << vx << " " << vy << " " << vz << " " << vr << endl;

	double vxg = 0, vyg = 0, vzg = 0;

	for(int i = 0; i < iter; i++)
	{
		ardrone.move3D(vx, vy, vz, vr);

		ardrone.getVelocity(&vxg, &vyg, &vzg);
	}
	
//	ardrone.getVelocity(&vx, &vy, &vz);	
//	cout << "\n	Velocity: " << " " << vx << " "<< vy << " "  << vz;
//	cout << "\n----------------------\n";		


	sleep(3);

	ardrone.landing();

	pthread_exit(NULL);
}

int main(int argc, char* argv[])
{

	try
	{

	    // Initialize
	    if (!ardrone.open()) {
       		 std::cout << "Failed to initialize." << std::endl;
       		 return -1;
   	     }

	   	 // Battery
		 std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]" << std::endl;

		ofstream pic_file;
		pic_file.open("./flight_data/flight_metric.txt");		  
		pic_file << " Test data\n";

		while (1) {

			// Key input
			int key = cv::waitKey(33);
			if (key == 0x1b) break;

			// Get an image
			cv::Mat image = ardrone.getImage();
		
			double vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
			int iter = 0;
			
			
			// Take off / Landing 
			if (key == ' ')
			{
				if (ardrone.onGround())
				{
	
					// Move
					cout << "Input vx:  vy:  vz:  vr:  iter  ";
					cin >> vx >> vy >> vz >> vr >> iter;

					ardrone.takeoff();
					cout << "Start" << endl;
					// Wait(secs) to stabilize, before commands
					sleep(10);

				//	ardrone.getVelocity(&vx, &vy, &vz);
				//	cout <<  " Values Passed: " << vx << " " << vy << " " << vz << " " << vr << endl;

					string path = "./flight_data/Obj_";
					time_t ti = time(0);
					struct tm * now = localtime(&ti);
					std::ostringstream oss;
					oss << now->tm_min  << now->tm_sec;
					string pic = path + oss.str() + ".jpg";		
							
					pic_file << pic << "\n";
					imwrite(pic, image);

                       			 pthread_t threads[NUM_THREADS];
                      			 int rc;
				 	 long t;	

					for(t = 0; t < NUM_THREADS; t++)
					{

						thread_data_array[t].vxt = vx;
						thread_data_array[t].vyt = vy;
						thread_data_array[t].vzt = vz;
						thread_data_array[t].vrt = vr;
						thread_data_array[t].itert = iter;

						rc = pthread_create(&threads[t], NULL, move,
								(void *) &thread_data_array[t]);
						if(rc){
							cout << "Error creating thread" << endl;
							exit(-1);
						}
					 }


				}
				else ardrone.landing();			
			}

			// Display the image
			cv::imshow("camera", image);
		}

		pic_file.close();
		ardrone.close();
	}	
	catch (const cv::Exception& ex)
	{

		std::cout << "Error: " << ex.what() << std::endl;

	}

	return 0;
}
