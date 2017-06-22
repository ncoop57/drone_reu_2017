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
		ardrone.move3D(vx, vy, vz, vr);
	pthread_exit(NULL);
}

int main(int argc, char* argv[])
{

	try
	{
//		ARDrone ardrone;

    // Initialize
    if (!ardrone.open()) {
        std::cout << "Failed to initialize." << std::endl;
        return -1;
    }

    // Battery
    std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]" << std::endl;

		while (1) {

			// Key input
			int key = cv::waitKey(1);
			if (key == 0x1b) break;

			// Get an image
			cv::Mat image = ardrone.getImage();

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

			// Move
			double vx = .0, vy = 0.0, vz = 0.0, vr = 1.0;
/*
			for(int i = 0; i < 5; i++)
			  	ardrone.move3D(vx, vy, vz, vr);
*/

					pthread_t threads[NUM_THREADS];
					int rc;
					long t;

					for(t = 0; t < NUM_THREADS; t++)
					{

						thread_data_array[t].vxt = vx;
						thread_data_array[t].vyt = vy;
						thread_data_array[t].vzt = vz;
						thread_data_array[t].vrt = vr;

						rc = pthread_create(&threads[t], NULL, move, 
								(void *) &thread_data_array[t]); 
						if(rc){
							cout << "Error creating thread" << endl;
							exit(-1);
						}
					} 

			// Display the image
			cv::imshow("camera", image);
		}

		ardrone.close();
	}	
	catch (const cv::Exception& ex)
	{

		std::cout << "Error: " << ex.what() << std::endl;

	}

	return 0;
}
