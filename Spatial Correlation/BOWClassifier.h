
#include<map>
#include<vector>
#include<string>
#include<iostream>
#include<ctime>
#include<cmath>
#include <fstream>
#include "ScaledImg.h"

#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>

#include <opencv2\features2d.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace std;
using namespace cv::ml;

typedef map<string, vector<string> > Dataset;

bool compareNoCase(string first, string second)
{
	int i = 0;
	while ((i < first.length()) && (i < second.length()))
	{
		if (tolower(first[i]) < tolower(second[i])) return true;
		else if (tolower(first[i]) > tolower(second[i])) return false;
		i++;
	}

	if (first.length() < second.length()) return true;
	else return false;
}

class BOW
{
public:
	BOW(const vector<string> &_class_list) : class_list(_class_list) {}

	void execute(const Dataset &filenames, bool isTrain = true) {
		int num_images = 0;
		vector<Mat> all_descriptors;
		vector<int> all_horiz_inds;
		vector<int> all_vert_inds;
		vector<int> class_sizes;
		int n;
		if (isTrain)
		{
			n = 150;
		}
		else
		{
			n = 50;
		}
		for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
		{
			for (int i = 0; i < c_iter->second.size(); i++)
			{
				Mat img = imread(String(c_iter->second[i].c_str()));
				cvtColor(img, img, cv::COLOR_BGR2GRAY);
				for (int j = 0, k = 1; k <= 4; k *= 2, j++) {

					Mat resized_img(256 / k, 256 / k, CV_8U);
					resize(img, resized_img, resized_img.size(), 0, 0, cv::INTER_AREA);
					ScaledImg sift_img(resized_img, j);
					sift_img.compute_dense_sift();
					all_descriptors.push_back(sift_img.descriptors);
					all_horiz_inds.insert(all_horiz_inds.end(), sift_img.horizRegionInds.begin(), sift_img.horizRegionInds.end());
					all_vert_inds.insert(all_vert_inds.end(), sift_img.vertRegionInds.begin(), sift_img.vertRegionInds.end());

				}
				if (i == n)
					break;


			}
			class_sizes.push_back(n);
		}
		cout << endl << " Done extracting descriptors";
		Mat descriptors, labels;
		vconcat(all_descriptors, descriptors);
		int k;
		string svm_file;
		if (isTrain) {
			cout << "\nTraining " << descriptors.rows << " descriptors" << endl;
			k = 250;
			kmeans(descriptors, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1), 3, KMEANS_PP_CENTERS, centers);
			cout << endl << "labels shape " << labels.rows << " x " << labels.cols;

			FileStorage f("word_cluster_centers.xml", FileStorage::WRITE);
			f << "centers" << centers;
			svm_file = "bow_train_features.svm";
		}
		else {
			load_model();
			k = centers.rows;
			svm_file = "bow_test_features.svm";
			labels.create(descriptors.rows, 1, CV_32S);

			BFMatcher matcher;
			vector<vector <DMatch> > matches;
			matcher.knnMatch(descriptors, centers, matches,1);
			cout << endl << "Assigning clusters";
			if ((int)matches.size() == descriptors.rows) 
            {
				for (int i = 0; i < labels.rows; i++) 
                {
					labels.at<int>(matches[i][0].queryIdx) = matches[i][0].trainIdx;
				}
			}
			else
            {
				cout << endl << " Matches for certain queries not found";
				exit(0);
			}
		}
		
		Dataset::const_iterator c_iter = filenames.begin();
		ofstream fout(svm_file.c_str() );
		cout << "\n Building SVM data";
		int s = 0, cl = 1, n_imgs = 0;
		vector<float> hist(28 * k, 0);
		
        for (int i = 0; i < all_descriptors.size(); i++) 
        {
			int num_desc = all_descriptors[i].rows;
			if (i % 3 == 0) 
            {

				for (int j = 0; j< 28 * k; j++) 
                {
					hist[j] = 0;
				}
			}
			for (int j = 0; j < num_desc; j++, s++) 
            {
				int scale = i % 3, horiz_index = 0, vert_index = 0;
				if (scale == 0)
				{
					horiz_index = all_horiz_inds[s] * k + labels.at<int>(s);
					vert_index = 8 * k + all_vert_inds[s] * k + labels.at<int>(s);
				}
				else if (scale == 1) 
                {
					horiz_index = 16 * k + all_horiz_inds[s] * k + labels.at<int>(s);
					vert_index = 20 * k + all_vert_inds[s] * k + labels.at<int>(s);
				}
				else 
                {
					horiz_index = 24 * k + all_horiz_inds[s] * k + labels.at<int>(s);
					vert_index = 26 * k + all_vert_inds[s] * k + labels.at<int>(s);

				}
				hist[horiz_index]++;
				hist[vert_index]++;
			}
			if (n_imgs == class_sizes[cl - 1]) 
            {
				cl++;
				n_imgs = 0;
			}
			if (i % 3 == 2) 
            {
				fout << cl;
				float sum = 0.0;
				for (int j = 0; j < 28 * k; j++)
					sum += hist[j];
				for (int j = 0; j < 28 * k; j++) {
					if (hist[j] != 0)
						fout << '\t' << (j + 1) << ':' << hist[j] / sum;
				}
				n_imgs++;
				fout << '\n';

			}


		}
		fout.close();

	}
	virtual void train(const Dataset &filenames)
	{
		execute(filenames);
	}

	int get_cluster_index(vector<float> descriptor)
	{
		int cluster = 0;
		float distance = 100000000.0;
		for (int i = 0; i<centers.rows; i++)
		{
			float desc_dist = 0.0;
			for (int j = 0; j<128; j++)
			{
				desc_dist += (centers.at<float>(i, j) - descriptor[j])*(centers.at<float>(i, j) - descriptor[j]);
			}

			if (desc_dist<distance)
			{
				distance = desc_dist;
				cluster = i;
			}
		}
		return cluster;
	}

	virtual string classify(const string &filename)
	{
		return 0;
	}

	virtual void load_model()
	{

		FileStorage f("word_cluster_centers.xml", FileStorage::READ);
		f["centers"] >> centers;
	}

	virtual void test(const Dataset &filenames)
	{
		execute(filenames, false);
	}

protected:
	vector<string> class_list;
	Mat centers;
};
