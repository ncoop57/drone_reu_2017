/*
 * drone.cpp
 *
 *  Created on: Jun 8, 2017
 *      Author: natha
 */

#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
//#include <unordered_map>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void matchFeatures(BFMatcher matcher, vector<DMatch>& good_matches, vector<KeyPoint>& prevKeypoints, vector<KeyPoint>& currKeypoints, Mat& prevDescriptors, Mat& currDescriptors);
void filter(vector<KeyPoint>& prevKeypoints, vector<KeyPoint>& currKeypoints, vector<DMatch>& good_matches, vector<DMatch> matches, Mat);
void detectObstacle(Mat, Mat&, vector<KeyPoint>&, vector<DMatch>);
void detectPlane(Mat frame, Mat& edges, vector<KeyPoint>& keypoints);

int main()
{

	VideoCapture cap(0);
	Ptr<SURF> detector = SURF::create(400);
	BFMatcher matcher;
	vector<DMatch> good_matches;
	Mat prevFrame, currFrame, grayFrame, prevDescriptors, currDescriptors;
	vector<KeyPoint> prevKeypoints, currKeypoints;

	if (!cap.read(prevFrame))
		return 0;

	cvtColor(prevFrame, grayFrame, CV_BGR2GRAY);
	detector->detect(grayFrame, prevKeypoints);
	detector->compute(grayFrame, prevKeypoints, prevDescriptors);

	for (;;)
	{

			if (!cap.read(currFrame))
					return 0;

			cvtColor(currFrame, grayFrame, CV_BGR2GRAY);
			detector->detect(grayFrame, currKeypoints);
			detector->compute(grayFrame, currKeypoints, currDescriptors);

			matchFeatures(matcher, good_matches, prevKeypoints, currKeypoints, prevDescriptors, currDescriptors);


			vector<KeyPoint> keypoints;
			for (size_t i = 0; i < good_matches.size(); i++)
				keypoints.push_back(currKeypoints[good_matches[i].trainIdx]);

			Mat edges, out;
			//subtract(currFrame, prevFrame, out);
			//detectObstacle(currFrame, edges, currKeypoints, good_matches);

			detectPlane(currFrame, edges, keypoints);

			Mat output;
			drawKeypoints(currFrame, keypoints, output, Scalar(255, 0, 0));
			//drawMatches(prevFrame, prevKeypoints, currFrame, currKeypoints, good_matches, output, Scalar(255, 0, 0), Scalar(255, 0, 0));
			imshow("Original", currFrame);
			imshow("Edges", edges);
			imshow("Output", output);

			char c = waitKey(30);

			if (c == 27) // Esc character
				break;

			currFrame.copyTo(prevFrame);
			prevKeypoints.clear();

			for (int i = 0; i < currKeypoints.size(); i++)
				prevKeypoints.push_back(currKeypoints[i]);
			currKeypoints.clear();
			currDescriptors.copyTo(prevDescriptors);

			good_matches.clear();

	}

}

void detectPlane(Mat frame, Mat& edges, vector<KeyPoint>& keypoints)
{

	//unordered_map<cv::KeyPoint, int> hashtable;

	frame.copyTo(edges);
	if (keypoints.size() <= 2)
		return;

	//Need to find: relation(of radius) to real world size
	double radius = 50;
	vector<vector<KeyPoint> > groups(keypoints.size());
	cout << "Vector Size: " << keypoints.size() << endl;

	for (size_t i = 0; i < keypoints.size(); i++)
	{

		bool ingroup = false;
		int groupId = 0;

		for (size_t k = 0; k < groups.size(); k++)
		{

			for (size_t j = 0; j < groups[i].size(); j++)
			{

				//Compare x and y coordinates seperately?
				if ( (keypoints[i].pt.x == groups[k][j].pt.x) && (keypoints[i].pt.y == groups[k][j].pt.y) )
				{
					ingroup = true;
					groupId = k;
					break;
				}

			}

		}

		if (ingroup)
		{

			for (size_t j = 0; j < keypoints.size(); j++ )
			{

				//Surely there is a distance function
				Point diff = keypoints[i].pt - keypoints[j].pt;
				float dist = sqrt(diff.x*diff.x + diff.y*diff.y);

				if(dist < radius)
				{
					//Check if in the same group; if so, continue
					bool same_gp = false;

					for (int m = 0; m < groups[groupId].size(); m++)
					{

						//Compare x and y coordinates separately?
						if ( (keypoints[j].pt.x == groups[groupId][m].pt.x) && (keypoints[j].pt.y == groups[groupId][m].pt.y) )
						{
							same_gp = true;
							continue;
						}

					}

					if(!same_gp)
					{

						groups[groupId].push_back(keypoints[j]);

					}

				}
			}

		}

		// If not in a group already
		else
		{

			vector<KeyPoint> single;
			single.push_back(keypoints[i]);
			groups.push_back(single);

			for (size_t j = 0; j < keypoints.size(); j++ )
			{

				if (i == j)
					continue;

				//Surely there is a distance function
				Point diff = keypoints[i].pt - keypoints[j].pt;
				float dist = std::sqrt(diff.x*diff.x + diff.y*diff.y);

				if(dist < radius)
				{
					//Most recent addition to groups
					groups[groups.size()-1].push_back(keypoints[j]);

				}

			}
		}
	}


	// Compute the centroid of each group
	vector<Point> average_in_groups;

	vector<Point> points;
	vector<vector<Point> > pt_groups(groups.size());

	for(size_t i = 0; i < groups.size(); i++)
	{
		 Point2f center(0,0);

		for(size_t j = 0; j < groups[i].size(); j++)
		{

			center.x += groups[i][j].pt.x;
			center.y += groups[i][j].pt.y;
			points.push_back(groups[i][j].pt);

		}

		center.x /= groups[i].size();
		center.y /= groups[i].size();
		average_in_groups.push_back(center);
		pt_groups.push_back(points);

	}

	vector<vector<Point> > hull(groups.size());
	for (size_t i = 0; i < hull.size(); i++)
	{

		convexHull(Mat(pt_groups[i]), hull[i], false);
		fillConvexPoly(edges, Mat(hull[i]), Scalar(255, 0, 0));
		cv::drawContours(edges, hull, i, cv::Scalar(255, 0, 0), 1, 8,
										std::vector<cv::Vec4i>(), 0, cv::Point());

	}



}


void detectObstacle(Mat frame, Mat& edges, vector<KeyPoint>& keypoints, vector<DMatch> good_matches)
{

	Mat temp, temp2;
	GaussianBlur(frame, temp, Size(3,3), 0, 0);
	Canny(temp, edges, 55, 100, 3, true);
	dilate(edges, edges, Mat(), Point(-1, -1), 3);


	vector<Vec4i> lines;
	HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10 );
	cvtColor(edges, edges, CV_GRAY2BGR);

	for( size_t i = 0; i < lines.size(); i++ )
	{
		Vec4i l = lines[i];
		line( edges, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,255), 3, CV_AA);
	}

	edges.copyTo(temp);
	edges.copyTo(temp2);

	try
	{

		if (good_matches.size() <= 0)
			return;

		int prev_Xavg = 0;
		int prev_Yavg = 0;

		int curr_Xavg = 0;
		int curr_Yavg = 0;

		for (size_t i = 0; i < good_matches.size(); i++)
		{

			Point prevPoint = Point((int)keypoints[good_matches[i].queryIdx].pt.x, (int)keypoints[good_matches[i].queryIdx].pt.y);
			Point currPoint = Point((int)keypoints[good_matches[i].trainIdx].pt.x, (int)keypoints[good_matches[i].trainIdx].pt.y);
			//Point((int)keypoints[good_matches[i].queryIdx].pt.x, (int)keypoints[good_matches[i].queryIdx].pt.y)
			if (((prevPoint.x < 0 || prevPoint.x > edges.cols) || (prevPoint.y < 0 || prevPoint.y > edges.rows)) && ((currPoint.x < 0 || currPoint.x > edges.cols) || (currPoint.y < 0 || prevPoint.y > edges.rows)))
				continue;

			prev_Xavg += keypoints[good_matches[i].queryIdx].pt.x;
			prev_Yavg += keypoints[good_matches[i].queryIdx].pt.y;

			curr_Xavg += keypoints[good_matches[i].trainIdx].pt.x;
			curr_Yavg += keypoints[good_matches[i].trainIdx].pt.y;

			/*Vec3b colour = edges.at<Vec3b>(Point((int)keypoints[good_matches[i].queryIdx].pt.x, (int)keypoints[good_matches[i].queryIdx].pt.y));
			if (!(colour[0] == 255 && colour[1] == 255 && colour[2] == 255))
			{


				//cout << "Previous - X: " << prevPoint.x << " Y: " << prevPoint.y << endl;
				//cout << "Current - X: " << currPoint.x << " Y: " << currPoint.y << endl;

				//Point((int)keypoints[good_matches[i].queryIdx].pt.x, (int)keypoints[good_matches[i].queryIdx].pt.y)
				if (((prevPoint.x < 0 || prevPoint.x > edges.cols) || (prevPoint.y < 0 || prevPoint.y > edges.rows)) && ((currPoint.x < 0 || currPoint.x > edges.cols) || (currPoint.y < 0 || prevPoint.y > edges.rows)))
					continue;

				int prev = floodFill(temp, Point((int)keypoints[good_matches[i].queryIdx].pt.x, (int)keypoints[good_matches[i].queryIdx].pt.y), Scalar(255, 0, 0));
				int curr = floodFill(temp2, Point((int)keypoints[good_matches[i].trainIdx].pt.x, (int)keypoints[good_matches[i].trainIdx].pt.y), Scalar(255, 0, 0));

				int ratio = curr / prev;

				if (ratio > 5 && ratio < 20)
				{

					cout << "Ratio: " << ratio << endl;
					floodFill(edges, keypoints[good_matches[i].queryIdx].pt, Scalar(0, 0, 255));

				}

			}*/

		}


		prev_Xavg /= good_matches.size();
		prev_Yavg /= good_matches.size();

		curr_Xavg /= good_matches.size();
		curr_Yavg /= good_matches.size();
		cout << "Avg X: " << curr_Xavg << " Avg Y: " << curr_Yavg << endl;

		Vec3b colour = edges.at<Vec3b>(Point(curr_Xavg, curr_Yavg));
		if (!(colour[0] == 255 && colour[1] == 255 && colour[2] == 255))
		{




			int prev = floodFill(temp, Point(prev_Xavg, prev_Yavg), Scalar(255, 0, 0));
			int curr = floodFill(temp2, Point(curr_Xavg, curr_Yavg), Scalar(255, 0, 0));

			circle(edges, Point(curr_Xavg, curr_Yavg), 15, Scalar(255, 0, 0), 5);


			double ratio = (double) curr / prev;
			if (ratio >= 2 && ratio < 20)
			{

				cout << "Ratio: " << ratio << endl;
				floodFill(edges, Point(curr_Xavg, curr_Yavg), Scalar(0, 0, 255));

			}

		}

	}
	catch (const cv::Exception& ex)
	{

		cout << "Error: " << ex.what() << endl;

	}

}

void matchFeatures(BFMatcher matcher, vector<DMatch>& good_matches, vector<KeyPoint>& prevKeypoints, vector<KeyPoint>& currKeypoints, Mat& prevDescriptors, Mat& currDescriptors)
{

	vector<DMatch> matches;
	matcher.match(prevDescriptors, currDescriptors, matches);
	filter(prevKeypoints, currKeypoints, good_matches, matches, prevDescriptors);

}

void filter(vector<KeyPoint>& prevKeypoints, vector<KeyPoint>& currKeypoints, vector<DMatch>& good_matches, vector<DMatch> matches, Mat prevDescriptors)
{

	double max_dist = 0;
	double min_dist = 100;

	for (size_t i = 0; i < matches.size(); i++)
	{

		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;

		if (dist > max_dist)
			max_dist = dist;

	}

	for (size_t i = 0; i < matches.size(); i++)
	{

		int curr = matches[i].trainIdx;
		int prev = matches[i].queryIdx;

		double ratio =  currKeypoints[curr].size / prevKeypoints[prev].size;
		if (matches[i].distance < 4 * min_dist && ratio > 1.2)
		{

			good_matches.push_back(matches[i]);

		}

	}

}
