#include <opencv2/core.hpp>
#include <opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <list>

#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <atlstr.h>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace boost::filesystem;


void thinningGuoHallIteration(cv::Mat& im, int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

	for (int i = 1; i < im.rows; i++)
	{
		for (int j = 1; j < im.cols; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
				(!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
			int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
			int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
			int N = N1 < N2 ? N1 : N2;
			int m = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

			if (C == 1 && (N >= 2 && N <= 3) & m == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}



void thinningGuoHall(cv::Mat& im)
{
	im /= 255;

	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningGuoHallIteration(im, 0);
		thinningGuoHallIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	im *= 255;
}





int main(char *argv[]) {
	string	ImgPath = "D:/Machine Learning/number/test_image/";
	//string ImgPath = argv[0];
	cout << ImgPath.length() << endl;
	int cmd;
	cout << "1: Threshold Number, 2: Skeleton Number, 3: Feed single image. Please input cmd..." << endl;
	cin >> cmd;
	path p = ImgPath;
	typedef vector<path> vec;
	vec v;
	copy(directory_iterator(p), directory_iterator(), back_inserter(v));
	sort(v.begin(), v.end());
	switch (cmd)	
	{
	case 1:
		try {
			int i = 0;
			int thresh;
			cout << "Please input threshold value... 1~255:" << endl;
			cin >> thresh;
			for (vec::const_iterator it(v.begin()); it != v.end(); ++it) {
				string ImgName = v[i].generic_string();
				Mat ori = imread(ImgName);
				Mat dst;
				
				if (ori.empty()) {
					cout << "No image loaded..." << endl;
					return -1;
				}
				threshold(ori, dst, thresh, 255, CV_THRESH_BINARY);
				ImgName = ImgName.substr(ImgPath.length());
				cv::imshow("Threshold img", dst);
				cv::waitKey(30);
				imwrite(ImgPath + "/Threshold/"+ImgName, dst);

				i++;
			}
		}
		catch (const filesystem_error& ex)
		{
			cout << ex.what() << '\n';
		}
		break;

	case 2:
		try {
			
			int i = 0;
			for (vec::const_iterator it(v.begin()); it != v.end(); ++it) {
				string ImgName = v[i].generic_string();
				Mat ori = imread(ImgName, CV_LOAD_IMAGE_GRAYSCALE);
				if (ori.empty()) {
					cout << "No image loaded..." << endl;
					return -1;
				}
				Mat skel;
				
				//cvtColor(ori, skel, CV_BGR2GRAY);
				skel = ori.clone();
				thinningGuoHall(skel);

				Mat imgLabel, stats, centroids;
				int *p;
				int nlabel = connectedComponentsWithStats(skel, imgLabel, stats, centroids, 8, CV_32S);
				if (nlabel > 2) {
					vector<int> area;
					for (int x = 0; x < skel.rows; x++) {
						p = imgLabel.ptr<int>(x);
						for (int y = 0; y < skel.cols; y++)
						{
							if (area.empty() || std::find(area.begin(), area.end(), stats.at<int>(p[y])) == area.end())
								area.push_back(stats.at<int>(p[y], CC_STAT_AREA));	
						}
					}
					sort(area.rbegin(), area.rend());
					int subMax, Max = area[0];
					for (int i = 0; i < area.size(); i++) {
						subMax = area[i];
						if (subMax < Max) break;
					}
					for (int x = 0; x < skel.rows; x++) {
						p = imgLabel.ptr<int>(x);
						for (int y = 0; y < skel.cols; y++) {
							if (stats.at<int>(p[y], CC_STAT_AREA) != Max && stats.at<int>(p[y], CC_STAT_AREA) != subMax)
								skel.at<uchar>(x, y) = 0;
						}
					}
					
				}
				threshold(skel, skel, 20, 255, THRESH_BINARY);
				
				cv::imshow("thin", skel);
				//imshow("Contour", black);
				ImgName = ImgName.substr(ImgPath.length());
				cout << ImgName << endl;
				cv::waitKey(10);
				imwrite(ImgPath + "/Thin_noNoise/"+ImgName, skel);

				i++;
			}

		}
		catch (const filesystem_error& ex)
		{
			cout << ex.what() << '\n';
		}


		cout << "Hit space for solving TSP ..." << endl;
		waitKey(0);
		FILE *file;
		//Py_SetProgramName("ImgSequence_tspSolver_AlL.py");
		Py_Initialize();
		file = fopen("ImgSequence_tspSolver_AlL.py", "r");
		cout << "Opening ImgSequence_tspSolver_AlL.py..." << endl;
		PyRun_SimpleFile(file, "ImgSequence_tspSolver_AlL.py");
		Py_Finalize();
		break;

	case 3:
		string ImgName = "D:/Machine Learning/number/4_3104.jpg";
		Mat ori = imread(ImgName, CV_LOAD_IMAGE_GRAYSCALE);
		if (ori.empty()) {
			cout << "No image loaded..." << endl;
			return -1;
		}
		Mat skel;
		imshow("ori", ori);
		waitKey(0);

		//cvtColor(ori, skel, CV_BGR2GRAY);
		ori.copyTo(skel);
		thinningGuoHall(skel);

		imshow("skel1", skel);

		Mat imgLabel, stats, centroids;
		int *p;
		int nlabel = connectedComponentsWithStats(skel, imgLabel, stats, centroids, 8, CV_32S);
		if (nlabel > 2) {
			vector<int> area;
			for (int x = 0; x < skel.rows; x++) {
				p = imgLabel.ptr<int>(x);
				for (int y = 0; y < skel.cols; y++)
				{
					if (area.empty() || std::find(area.begin(), area.end(), stats.at<int>(p[y])) == area.end())
						area.push_back(stats.at<int>(p[y], CC_STAT_AREA));
				}
			}
			sort(area.rbegin(), area.rend());
			int subMax, Max = area[0];
			for (int i = 0; i < area.size(); i++) {
				subMax = area[i];
				if (subMax < Max) break;
			}
			for (int x = 0; x < skel.rows; x++) {
				p = imgLabel.ptr<int>(x);
				for (int y = 0; y < skel.cols; y++) {
					if (stats.at<int>(p[y], CC_STAT_AREA) != Max && stats.at<int>(p[y], CC_STAT_AREA) != subMax)
						skel.at<uchar>(x, y) = 0;
				}
			}

		}

		threshold(skel, skel, 20, 255, THRESH_BINARY);

		cv::imshow("thin", skel);
		waitKey(0);
	}

	
	

	return 0;

}