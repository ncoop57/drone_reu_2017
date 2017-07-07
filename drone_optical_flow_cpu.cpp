#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "ardrone/ardrone.h"

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <ctype.h>
#include <pthread.h>

using namespace cv;
using namespace std;

#define NUM_THREADS 1

/* Helper functions used for obstacle detection */
void getPoints(Mat, Mat, vector<Point2f>&, vector<Point2f>&);
void getSegmentation(Ptr<cv::ximgproc::segmentation::GraphSegmentation>, Mat, Mat&);
void getGroups(Mat, Mat, vector<vector<Point2f> >&, vector<vector<Point2f> >&, vector<Point2f>, vector<Point2f>);

/* Function used for obstacle avoidance */
void getMovement(Mat&, Mat, vector<vector<Point2f> >, vector<vector<Point2f> >, double&, double&, double&, double&);

ARDrone ardrone;
int number;
int FLAG = 0;
const int MAX_COUNT = 500;
int frame_index = 0;
int MAX_DIST = 25;
int MIN_DIST = 10;
int MAX_RATIO = 5;
double MIN_RATIO = 1.2;
std::ostringstream fss;

// Struct used to pass arguments to Pthread
typedef struct{
	double vxt;
	double vyt;
	double vzt;
	double vrt;
}thread_data;

thread_data thread_data_array[NUM_THREADS];

/* Variables used for measuring FPS */
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

/* Pthread function for object avoidance. */
/* This Pthread function recieves the x, y, z, and rotation parameters of UAV
 * and moves the UAV accordingly, for 5 iterations.
 * @param threadarg struct used to pass x, y, z, and rotation  parameters by value
*/
void *move(void *threadarg)
{
//   thread_data *my_data;

//	my_data = (thread_data *) threadarg;



	for(int i = 0; i < 5; i++)
			ardrone.move3D(0.0, 0, 0, 0);
/*
	if (my_data->vyt != 0 || my_data->vzt != 0)
	{
		cout << "Stop-Turn..." << endl;
		for(int i = 0; i < 5; i++)
			ardrone.move3D(0, my_data->vyt, 0, 0);
	}
	else
		for(int i = 0; i < 5; i++)
			ardrone.move3D(0.1, 0, 0, 0);
*/
	pthread_exit(NULL);

}

int main()
{

	if (!ardrone.open())
			return -1;

	/* Open file for numbering */
	std::fstream numberfile("./flight_data/count.txt", std::ios_base::in);
	numberfile >> number;
	cout << "File number: " << number << endl;
	numberfile.close();

	/* Used to store FPS and write to file */
	double storage = 0;

	/* Segmentation variable */
	Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg = cv::ximgproc::segmentation::createGraphSegmentation(0.5, 500, 50);
	namedWindow("Original", 1);
	createTrackbar("Max Ratio", "Original", &MAX_RATIO, 50);
	createTrackbar("Max Distance", "Original", &MAX_DIST, 100);

	vector<Point2f> prev_points;
	vector<Point2f> curr_points;

	/* Read in image, resize image, and convert to Gray scale */
	Mat original, prev_frame, curr_frame;
	original = ardrone.getImage();
	resize(original, original, Size(200, 200));
	cvtColor(original, prev_frame, CV_RGB2GRAY);
	

	for (;;)
	{

		clock_gettime(CLOCK_MONOTONIC, &start);

		std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]\r" << std::flush;

		/* Read in image, resize image, and convert to Gray scale */
		original = ardrone.getImage();
		resize(original, original, Size(200, 200));
		cvtColor(original, curr_frame, CV_RGB2GRAY);

		getPoints(prev_frame, curr_frame, prev_points, curr_points);

		Mat segmentation;
		getSegmentation(seg, curr_frame, segmentation);

		vector<vector<Point2f> > prev_groups, curr_groups;
		getGroups(original, segmentation, prev_groups, curr_groups, prev_points, curr_points);

		double vx = 1.0, vy = 0.0, vz = 0.0, vr = 0.0;
		getMovement(original, segmentation, prev_groups, curr_groups, vx, vy, vz, vr);

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

		if (FLAG == 1)
		{

			/* Write pic and directions to file*/
			string path = "/Obj_";
			time_t ti = time(0);
			struct tm * now = localtime(&ti);
			std::ostringstream oss;
			oss << fss.str() << path << "_" << now->tm_min << "_" << now->tm_sec << ".jpg";
			imwrite(oss.str(), original);

			/* Open file for input */
			ofstream pic_file;
			std::ostringstream pss;
			pss << fss.str() << "/flight_metric.txt";
			pic_file.open(pss.str().c_str());
			pic_file << " Test data\n";

			pss << "Left/Right: " << vy << "\tUp/Down: " << vz << endl;
			pss << "Time: " << elapsed << endl;
			pss << "FPS: " << storage << endl;
			pic_file << pss.str() << endl << endl;
			FLAG = 0;

		}

		char key = waitKey(30);

		// Take off / Landing
		if (key == ' ')
		{
			if (ardrone.onGround())
			{
				ardrone.takeoff();
				
				fss.str("");
				fss.clear();
				fss << "./flight_data/Test_" << number;
				mkdir(fss.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

				cout << "Start" << endl;
				// Wait(secs) to stabilize, before commands
				sleep(10);
				ardrone.move3D(0, 0, -0.05, 0);
				sleep(3);
				cout << "End" << endl;
			}
			else
			{ 
				ardrone.landing();
				number++;
			}

		}

  		// Change camera
  		static int mode = 0;
  		if (key == 'c') ardrone.setCamera(++mode % 4);	

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

	std::fstream computerfile("./flight_data/count.txt");
	computerfile << number << std::flush;
	computerfile.close();

	cout << "Average FPS: " << avgfps() << endl;

	return 0;

}

void getPoints(Mat prev_frame, Mat curr_frame, vector<Point2f>& prev_points, vector<Point2f>& curr_points)
{

	if (frame_index % 5 == 0)
		goodFeaturesToTrack(prev_frame, prev_points, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
		
	vector<uchar> status;
   vector<float> err;
	calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, curr_points, status, err);

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


void getSegmentation(Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg, Mat curr_frame, Mat& curr_output_image)
{

	//

	Mat curr_input, curr_output;

	double mins, maxs;

	// Segmentation of current frame
	seg->processImage(curr_frame, curr_output);

	minMaxLoc(curr_output, &mins, &maxs);

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

void getGroups(Mat original, Mat curr_output_image, vector<vector<Point2f> >& prev_groups, vector<vector<Point2f> >& curr_groups, vector<Point2f> prev_points, vector<Point2f> curr_points)
{

	vector<Point2f> p_points;
	vector<Point2f> c_points;

	for (unsigned int i = 0; i < curr_points.size(); i++)
	{

		double dist = norm(prev_points[i] - curr_points[i]);
		if (dist < MAX_DIST && dist > MIN_DIST)
		{

			circle( original, curr_points[i], 5, Scalar(0,0,255), -1, 10);
			p_points.push_back(prev_points[i]);
			c_points.push_back(curr_points[i]);

		}

	}

	/* Cluster points into groups by color */
	for(unsigned int i = 0; i < c_points.size(); i++)
	{

		int FLAG2 = 0;

		for(unsigned int j = 0; j < curr_groups.size(); j++)
			if(curr_output_image.at<Vec3b>(c_points[i]) == 
				curr_output_image.at<Vec3b>(curr_groups[j][0]))
			{

				prev_groups[j].push_back(p_points[i]);
				curr_groups[j].push_back(c_points[i]);
				FLAG2 = 1;

			}
		
		if( curr_groups.size() == 0 || FLAG2 == 0);
		{

			std::vector<cv::Point2f> prev_cluster;
			std::vector<cv::Point2f> curr_cluster;
			prev_cluster.push_back(p_points[i]);
			curr_cluster.push_back(c_points[i]);
			prev_groups.push_back(prev_cluster);
			curr_groups.push_back(curr_cluster);

		}
			
	}

}

void getMovement(Mat& frame, Mat segmentation, vector<vector<Point2f> > prev_groups, vector<vector<Point2f> > curr_groups, double& vx, double& vy, double& vz, double& vr)
{

	for (unsigned int i = 0; i < curr_groups.size(); i++)
	{


		
		// Used to approximate countours to polygons + get bounding rects
		std::vector<std::vector<cv::Point2f> > hull(2);
		cv::convexHull(prev_groups[i], hull[0], false);
		cv::convexHull(curr_groups[i], hull[1], false);
		cv::Rect boundRect;

		double prev_area = cv::contourArea(hull[0]);
		double curr_area = cv::contourArea(hull[1]); 

		double ratio = curr_area / prev_area;

		if (ratio > 1)
//			cout << "Ratio: " << ratio << endl;

		if (ratio > MIN_RATIO && ratio < MAX_RATIO)
		{

			try
			{

				if (curr_groups[i][0].x > segmentation.rows || curr_groups[i][0].y > segmentation.cols
					 || curr_groups[i][0].x < 0 || curr_groups[i][0].y < 0)
					continue;

				floodFill(segmentation, curr_groups[i][0], cv::Scalar(250, 250, 250), &boundRect);
				int area = boundRect.width * boundRect.height;

				if (area > 22500)
					continue;

			}
			catch(const cv::Exception& ex)
			{

				continue;

			}

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
			FLAG = 1;

			
		}/*/**/

	}

	if(vy > 0)
		putText(frame, "Left", cv::Point(frame.cols/2,frame.rows/2), FONT_HERSHEY_DUPLEX, 0.25, cv::Scalar(10,200,10), 1);
	if(vy < 0 )
		putText(frame, "Right", cv::Point(frame.cols/2,frame.rows/2), FONT_HERSHEY_DUPLEX, 0.25, cv::Scalar(10,200,10), 1);

	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

}


