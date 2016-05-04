#pragma once
#include <opencv2\core.hpp>
#include<opencv2\imgproc.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <vector>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class ScaledImg
{
public:
	Mat img;
	Mat descriptors;
	int downscale;
	int grid_size;
	int region_width;
	void set_grid_size() {
		grid_size = 4 * (4 - downscale);

	}
	vector<KeyPoint> kps;
	vector<int> horizRegionInds;
	vector<int> vertRegionInds;
	int stride;
	void set_stride() {
		stride = grid_size / 2;
	}
	void set_region_width() {
		double x = pow(2, 3 - downscale);
		region_width = (int)round(x);
	}
	void set_keypoints() {
		if (!img.empty() ) {
			for (int y = grid_size; y < img.rows - grid_size; y+=stride) {
				for (int x = grid_size; x < img.cols - grid_size; x+=stride) {
					KeyPoint k((float)x, (float)y, (float)grid_size);
					kps.push_back(k);
					vertRegionInds.push_back(y / (img.rows / region_width));
					horizRegionInds.push_back(x / (img.cols / region_width));

				}
			}
		}
	}
	void set_descriptors() {
		Ptr<SIFT> sift = SIFT::create();
		sift->detectAndCompute(img, Mat(), kps, descriptors, true);

	}
	ScaledImg(Mat image, int scale) {
		img = image;
		downscale = scale;
		set_grid_size();
		set_stride();
		set_region_width();
		//set_keypoints();
		//set_descriptors();
	}
	void compute_dense_sift() {
		set_keypoints();
		set_descriptors();
	}

};