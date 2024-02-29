#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat imgOrg, imgGray, imgCanny,imgGauss,imgDil,imgProcess,imgWarp,imgCrop,imgTh,IG;
vector<Point> initialPoints;
vector<Point> docPoints;
float w = 420, h = 740;


Mat process(Mat image)
{
	cvtColor(image, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgGauss, Size(3, 3), 3);
	Canny(imgGauss, imgCanny, 100, 150);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	dilate(imgCanny, imgDil, kernel);
	return imgDil;

}

vector<Point> getContours(Mat image)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;

	findContours(image, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	vector<Point> biggest;
	int maxArea=0;

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		//cout << area << endl;

		string objType;

		if (area > 1000)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
			//cout << conPoly[i].size() << endl;

			if (area > maxArea && conPoly[i].size() == 4)
			{
				biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
				maxArea = area;
				drawContours(imgOrg, conPoly, i, Scalar(255, 0, 255), 3);
			}

			//drawContours(imgOrg, conPoly, i, Scalar(255, 0, 255), 2);
		}
	}
	return biggest;
}

void drawPoints(vector<Point> points, Scalar color)
{
	for (int i = 0; i < points.size(); i++)
	{
		circle(imgOrg, points[i], 7, color, FILLED);
		putText(imgOrg, to_string(i), points[i], FONT_HERSHEY_PLAIN, 2, color, 2);
	}
}

vector<Point> reorder(vector<Point> points)
{
	vector<Point> newPoints;
	vector<int>  sumPoints, subPoints;

	for (int i = 0; i < 4; i++)
	{
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);
	}

	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); //0
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //1
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //2
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); //3

	return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h)
{
	Point2f src[4] = { points[0],points[1],points[2],points[3] };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	return imgWarp;
}

void main()
{
	string path = "./receipt.png";
	imgOrg = imread(path);
	resize(imgOrg, imgOrg, Size(), 0.3, 0.3);
	//cout << imgOrg.size() << endl;

	//preprocessing
	imgProcess = process(imgOrg);

	//rectangular contour
	initialPoints = getContours(imgProcess);
	//cout << initialPoints << endl;
	//drawPoints(initialPoints, Scalar(0, 0, 255));
	docPoints = reorder(initialPoints);
	drawPoints(docPoints, Scalar(0, 0, 255));

	//warp
	imgWarp = getWarp(imgOrg, docPoints, w, h);

	//crop
	int cropVal = 10;
	Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
	imgCrop = imgWarp(roi);

	//thresh
	cvtColor(imgCrop, IG, COLOR_BGR2GRAY);
	threshold(IG, imgTh, 0, 255, THRESH_BINARY | THRESH_OTSU);

	imshow("image", imgOrg);
	//imshow("process image", imgProcess);
	//imshow("warp image", imgWarp);
	//imshow("crop image", imgCrop);
	imshow("threshold image", imgTh);
	waitKey(0);

}