#include <map>
#include <dirent.h>
#include <iostream>
#include "BOWClassifier.h"

using namespace cv;
using namespace std;
typedef map<string, vector<string> > Dataset;

vector<string> files_in_directory(const string &directory, bool prepend_directory = false)
{
	vector<string> file_list;
	DIR *dir = opendir(directory.c_str());
	if (!dir)
		throw std::string("Can't find directory " + directory);

	struct dirent *dirent;
	while ((dirent = readdir(dir)))
		if (dirent->d_name[0] != '.')
			file_list.push_back((prepend_directory ? (directory + "/") : "") + dirent->d_name);
	closedir(dir);
	return file_list;
}

int main(int argc, char **argv)
{
	try
	{

		if (argc < 2)
			throw string("Insufficent number of arguments");

		string mode = argv[1];
		Dataset filenames;

		vector<string> class_list = files_in_directory("C:/"+mode);
		for (vector<string>::const_iterator c = class_list.begin(); c != class_list.end(); ++c)
			filenames[*c] = files_in_directory("C:/"+mode + "/" + *c, true);

		BOW bowClassifier(class_list);

		if (mode == "train")
		{
			bowClassifier.train(filenames);
		}
		else if (mode == "test")
		{
			bowClassifier.test(filenames);
		}
		else
			throw std::string("unknown mode!");

	}
	catch (const string &err) {
		cerr << "Error: " << err << endl;
	}
}
