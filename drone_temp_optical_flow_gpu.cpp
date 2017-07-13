#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include "ardrone/ardrone.h"

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <ctype.h>
#include <pthread.h>

using namespace cv;
using namespace std;
using namespace cv::cuda;


/* Helper functions used for obstacle detection */
void getPoints(Mat, Mat, vector<Point2f>&, vector<Point2f>&);
void getTemplate(Mat, Mat, vector<Point2f>&, vector<Point2f>&, vector<Point2f>&, vector<Point2f>&);
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
cv::RNG rng(12345);

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

int main()
{

	bool flag = false;

	if (!ardrone.open())
			return -1;

	VideoWriter writer;

	/* Open file for numbering */
	std::fstream numberfile("./flight_data/count.txt", std::ios_base::in);
	numberfile >> number;
	cout << "File number: " << number << endl;
	numberfile.close();

	/* Used to store FPS and write to file */
	double storage = 0;

	/* Segmentation variable */
	Ptr<cv::ximgproc::segmentation::GraphSegmentation> seg = cv::ximgproc::segmentation::createGraphSegmentation(0.5, 500, 50);
	/*namedWindow("Original", 1);
	createTrackbar("Max Ratio", "Original", &MAX_RATIO, 50);
	createTrackbar("Max Distance", "Original", &MAX_DIST, 100);*/

	vector<Point2f> prev_points;
	vector<Point2f> curr_points;
	vector<Point2f> prev_expand;
	vector<Point2f> curr_expand;

	/* Read in image, resize image, and convert to Gray scale */
	Mat original, roi, prev_frame, curr_frame, features;
	original = ardrone.getImage();
	cv::resize(original, original, Size(200, 200));
	Rect rect((int) (original.rows / 2 - (original.rows * 0.67) / 2), (int) (original.cols / 2 - (original.cols * 0.67) / 2), (int) (original.rows * 0.67), (int) (original.cols * 0.67));
	roi = original(rect);
	cv::cvtColor(roi, prev_frame, CV_RGB2GRAY);
	
	

	for (;;)
	{

		clock_gettime(CLOCK_MONOTONIC, &start);

		std::cout << "Battery = " << ardrone.getBatteryPercentage() << "[%]\r" << std::flush;

		/* Read in image, resize image, and convert to Gray scale */
		original = ardrone.getImage();
		cv::resize(original, original, Size(200, 200));
		Rect rect1((int) (original.rows / 2 - (original.rows * 0.67) / 2), (int) (original.cols / 2 - (original.cols * 0.67) / 2), (int) (original.rows * 0.67), (int) (original.cols * 0.67));
		roi = original(rect1);
		cv::cvtColor(roi, curr_frame, CV_RGB2GRAY);

		getPoints(prev_frame, curr_frame, prev_points, curr_points);
		getTemplate(prev_frame, curr_frame, prev_points, curr_points, prev_expand, curr_expand);
		
		Mat segmentation;
		getSegmentation(seg, curr_frame, segmentation);
		
		vector<vector<Point2f> > prev_groups, curr_groups;
		roi.copyTo(features);
		getGroups(features, segmentation, prev_groups, curr_groups, prev_expand, curr_expand);

		double vx = 1.0, vy = 0.0, vz = 0.0, vr = 0.0;
		getMovement(roi, segmentation, prev_groups, curr_groups, vx, vy, vz, vr);

		/* Calculate individual FPS */
		storage = avgfps();
		if (flag)
		{

			if (FLAG == 1)
			{

				/* Write pic and directions to file */
				string path = "/Obj_";
				time_t ti = time(0);
				struct tm * now = localtime(&ti);
				std::ostringstream oss;
				oss << fss.str() << path << "_" << now->tm_min << "_" << now->tm_sec << ".jpg";
				imwrite(oss.str(), roi);
				std::ostringstream segme;
				segme << fss.str() << path << "_" << now->tm_min << "_" << now->tm_sec << "_" << "seg" << ".jpg";
				imwrite(segme.str(), features);

				/* Open file for input  */
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

		
			
			writer << roi;

		}

		char key = waitKey(30);

		// Take off / Landing
		if (key == ' ')
		{
			if (!flag)
			{

				flag = true;
//				ardrone.takeoff();

				
				fss.str("");
				fss.clear();
				fss << "./flight_data/Temp_Experiment_" << number;
				mkdir(fss.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

				std::ostringstream filename;
				filename << fss.str() << "/video.avi";
				int fcc = CV_FOURCC('X', 'V', 'I', 'D');
				int fps = avgfps();
				Size frameSize(roi.cols, roi.rows);
			
				writer.open(filename.str(), fcc, fps, frameSize);
				

				cout << "Start" << endl;
				// Wait(secs) to stabilize, before commands
				//ardrone.move3D(0, 0, 0, 0);
				//sleep(10);


			}
			else
			{ 

				flag = false;
//				ardrone.landing();
				writer.release();
				number++;
				cout << "End" << endl;
			}

		}

  		// Change camera
  		static int mode = 0;
  		if (key == 'c') ardrone.setCamera(++mode % 4);

		imshow("Original", roi);
		imshow("Segment", segmentation);
		imshow("Features", features);

		curr_frame.copyTo(prev_frame);
		prev_points.clear();
		prev_points = curr_points;

		prev_expand.clear();
		curr_expand.clear();


		if (key == 27) // Esc key
			break;

		frame_index++;
	
		/* Drone Movement - move forward unless object detected */
/*		if (vy != 0)
		{

			for(int i = 0; i < 10; i++)
				if (vy < 0)
					ardrone.move3D(0, -0.1, 0, 0);
				else
					ardrone.move3D(0, 0.1, 0, 0);

		}
*/		//else
			//ardrone.move3D(0.3, 0, 0, 0);

		
	}

	ardrone.close();

	std::fstream computerfile("./flight_data/count.txt");
	computerfile << number << std::flush;
	computerfile.close();

	cout << "Average FPS: " << avgfps() << endl;

	return 0;

}
static void download(const GpuMat& d_mat, vector<Point2f>& vec)
{

	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)& vec[0]);
	d_mat.download(mat);

}

void getPoints(Mat prev_frame, Mat curr_frame, vector<Point2f>& prev_points, vector<Point2f>& curr_points)
{

	GpuMat d_prev_frame(prev_frame), d_curr_frame(curr_frame), d_prev_points(prev_points), d_curr_points;


	if (frame_index % 5 == 0)
	{

		Ptr<CornersDetector> detector = createGoodFeaturesToTrackDetector(d_prev_frame.type(), MAX_COUNT, 0.01, 10);
		detector->detect(d_prev_frame, d_prev_points);
		download(d_prev_points, prev_points);

	}
	
	Ptr<cuda::SparsePyrLKOpticalFlow> lk = cuda::SparsePyrLKOpticalFlow::create();

	lk->calc(d_prev_frame, d_curr_frame, d_prev_points, d_curr_points, GpuMat());
	download(d_curr_points, curr_points);

}

void getTemplate(Mat prev_frame, Mat curr_frame, vector<Point2f>& prev_points, vector<Point2f>& curr_points, vector<Point2f>& prev_expand, vector<Point2f>& curr_expand)
{

	Mat sample;
	curr_frame.copyTo(sample);
	for (unsigned int i = 0; i < curr_points.size(); i++)
		circle(sample, curr_points[i], 6, Scalar(20, 25, 200), -1);

	Mat result;

/* Template Matching */
	for(unsigned int i = 0; i < curr_points.size(); i++)
	{
		/* Draw template around previous feature */
		cv::Rect section1(prev_points[i].x - 15, prev_points[i].y - 15, 30, 30); 

		bool is_inside = (section1 & cv::Rect(0, 0, prev_frame.cols, prev_frame.rows)) == section1;
		if(!is_inside)
			continue;
		Mat PrevTemp = prev_frame(section1);

		double TMmin = 2, Scalemin = 2;
		for(double scale = 1; scale < 2; scale += 0.1)
		{

			/* Scale Previous feature */			
			Mat new_PrevTemp;
			cv::resize(PrevTemp, new_PrevTemp, cvSize(0, 0), scale, scale);

			/* Draw template around current feature */
			cv::Rect section2(curr_points[i].x - 15 * scale , curr_points[i].y - 15 * scale,
									  30 * scale, 30 * scale);

			bool is_inside2 = (section2 & cv::Rect(0, 0, curr_frame.cols, curr_frame.rows)) == section2;
			if(!is_inside2)
				continue;
			Mat CurrTemp = curr_frame(section2);

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
	
		}
		
		if(Scalemin > 1.06 && TMmin < 0.8)
		{
			prev_expand.push_back( prev_points[i]);
			curr_expand.push_back( curr_points[i]);
		}
	}

//	prev_points.swap(prev_hold);
//	curr_points.swap(curr_hold);
	
	for (unsigned int i = 0; i < curr_expand.size(); i++)
		circle(sample, curr_expand[i], 6, Scalar(200, 25, 20), -1);

	imshow("Expanding", sample);

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

			
			p_points.push_back(prev_points[i]);
			c_points.push_back(curr_points[i]);
		//cout << "X: " << c_points[i].x << " Y: " << c_points[i].y << endl;
			circle( original, curr_points[i], 5, curr_output_image.at<Vec3b>(Point(curr_points[i].x, curr_points[i].y)), -1, 10);

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

	/* Compute midpoint for the cluster */
	vector<cv::Point> curr_midpoints;
	vector<cv::Point> prev_midpoints;
	
	for(unsigned int d = 0; d < curr_groups.size(); d++)
	{
		
		double cx = 0, cy = 0, px = 0, py = 0;
		for(unsigned int p = 0; p < curr_groups[d].size(); p++)
		{

			cx += curr_groups[d][p].x;
			cy += curr_groups[d][p].y;

			px += prev_groups[d][p].x;
			py += prev_groups[d][p].y;

		}

		cx /= curr_groups[d].size();
		cy /= curr_groups[d].size();
		px /= prev_groups[d].size();
		py /= prev_groups[d].size();

		Point c_mid(cx, cy);
		Point p_mid(px, py);
	
		
        /* Add the cluster and midpoint to their respective containers */
//		if (curr_cluster.size() > 2)
		{

			curr_midpoints.push_back(c_mid);
			prev_midpoints.push_back(p_mid);
		}
	}

	for (unsigned int i = 0; i < curr_groups.size(); i++)
	{

		vector<Point> temp_p_group, temp_c_group;
		for (unsigned int j = 0; j < curr_groups[i].size(); j++)
		{

			temp_p_group.push_back(prev_groups[i][j]);
			temp_c_group.push_back(curr_groups[i][j]);

		}
		
		// Used to approximate countours to polygons + get bounding rects
		std::vector<std::vector<cv::Point> > hull(2);
		cv::convexHull(temp_p_group, hull[0], false);
		cv::convexHull(temp_c_group, hull[1], false);
		cv::Rect boundRect;

		double prev_area = cv::contourArea(hull[0]);
		double curr_area = cv::contourArea(hull[1]); 

		double ratio = curr_area / prev_area;

		if (ratio > MIN_RATIO && ratio < MAX_RATIO)
		{

			try
			{

				if (curr_groups[i][0].x > segmentation.rows || curr_groups[i][0].y > segmentation.cols
					 || curr_groups[i][0].x < 0 || curr_groups[i][0].y < 0)
					continue;

				floodFill(segmentation, curr_midpoints[i], cv::Scalar(250, 250, 250), &boundRect);
				int area = boundRect.width * boundRect.height;

				if (area > (double) (frame.rows * frame.cols) / 2)
					continue;

			}
			catch(const cv::Exception& ex)
			{

				continue;

			}
	 		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			// Used to approximate countours to polygons + get bounding rect			
			cv::rectangle(frame, boundRect.tl(), boundRect.br(),
							 color, 2, 8, 0);

			drawContours(frame, hull, 1, color, 1, 8, vector<Vec4i>(), 0, Point());

			// Check which quadrant(s) rectangle is in
			cv::Point top_left(boundRect.tl().x, boundRect.tl().y);
			cv::Point top_right(boundRect.tl().x + boundRect.width, boundRect.tl().y);
			cv::Point bot_left(boundRect.tl().x, boundRect.tl().y + boundRect.height);
			cv::Point bot_right(boundRect.tl().x + boundRect.width, boundRect.tl().y + boundRect.height);

			vy += min(1.0, 0.1 * (top_left.x - frame.cols / 2.0));
			vy += max(-1.0, 0.1 * (top_right.x - frame.cols / 2.0));
			FLAG = 1;

			
		}/**/

	}

	if(vy > 0)
		putText(frame, "Left", cv::Point(frame.cols/2,frame.rows/2), FONT_HERSHEY_DUPLEX, 0.25, cv::Scalar(10,200,10), 1);
	if(vy < 0 )
		putText(frame, "Right", cv::Point(frame.cols/2,frame.rows/2), FONT_HERSHEY_DUPLEX, 0.25, cv::Scalar(10,200,10), 1);

	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

}


