#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc.hpp>
#include "ardrone/ardrone.h"

#include <iostream>
#include <ctype.h>
#include <pthread.h>

using namespace cv;
using namespace std;

#define NUM_THREADS 1

ARDrone ardrone;
const int MAX_COUNT = 500;
int frame_index = 0;
int MAX_DIST = 25;
int MIN_DIST = 10;
int MAX_RATIO = 5;
double MIN_RATIO = 1.2;

// Struct used to pass arguments to Pthread
typedef struct{
	double vxt;
	double vyt;
	double vzt;
	double vrt;
}thread_data;

thread_data thread_data_array[NUM_THREADS];


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

int main()
{

	if (!ardrone.open())
			return -1;

	/* Used to store FPS and write to file */
	double storage = 0;

	/* Segmentation variable */
	Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg = cv::ximgproc::segmentation::createGraphSegmentation(0.5, 500, 50);
	namedWindow("Original", 1);
	createTrackbar("Max Ratio", "Original", &MAX_RATIO, 50);
	createTrackbar("Max Distance", "Original", &MAX_DIST, 100);

	vector<Point2f> prev_points;
	vector<Point2f> curr_points;

	Mat original, prev_frame, curr_frame;
	original = ardrone.getImage();
	cvtColor(original, prev_frame, CV_RGB2GRAY);
	

	for (;;)
	{

		original = ardrone.getImage();
		cvtColor(original, curr_frame, CV_RGB2GRAY);

//		getPoints(prev_frame, curr_frame, prev_points, curr_points);

		Mat segmentation;
//		getSegmentation(seg, curr_frame, segmentation);

		vector<vector<Point2f> > prev_groups, curr_groups;
//		getGroups(original, segmentation, prev_groups, curr_groups, prev_points, curr_points);

		double vx = 1.0, vy = 0.0, vz = 0.0, vr = 0.0;
//		getMovement(original, segmentation, prev_groups, curr_groups, vx, vy, vz, vr);

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

		char key = waitKey(30);

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

/*
		for (int i = 0; i < curr_points.size(); i++)
			circle( original, curr_points[i], 5, Scalar(0,0,0), -1, 10);
*/
		imshow("Original", original);
		//imshow("Segmentation", segmentation);

		curr_frame.copyTo(prev_frame);
		prev_points.clear();
		prev_points = curr_points;

		if (key == 27) // Esc key
			break;

		frame_index++;
		
	}

	

	ardrone.close();
	cout << "Average FPS: " << avgfps() << endl;

	return 0;

}
