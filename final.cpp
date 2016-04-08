#include "CImg.h"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include <vector>
#include <Sift.h>
#include <sys/types.h>
#include <dirent.h>
#include <map>
#include <numeric>
#include <opencv2/core.hpp>


typedef CImg<double> Image;

//Use the cimg namespace to access the functions easily
using namespace cimg_library;
using namespace std;
using namespace cv;

typedef map<string, vector<string> > Dataset;


#include <BOW.h>

// Figure out a list of files in a given directory.
//
vector<string> files_in_directory(const string &directory, bool prepend_directory = false)
{
    vector<string> file_list;
    DIR *dir = opendir(directory.c_str());
    cout<<"Works";
    if(!dir)
        throw std::string("Can't find directory " + directory);
    
    struct dirent *dirent;
    while ((dirent = readdir(dir)))
        if(dirent->d_name[0] != '.')
            file_list.push_back((prepend_directory?(directory+"/"):"")+dirent->d_name);
    
    closedir(dir);
    return file_list;
}

int main(int argc, char **argv)
{
    try
    {
        
        if(argc < 2)
            throw string("Insufficent number of arguments");
        
        string mode = argv[1];
        
        // Scan through the "train" or "test" directory (depending on the
        //  mode) and builds a data structure of the image filenames for each class.
        Dataset filenames;
        
        vector<string> class_list = files_in_directory(mode);
        for(vector<string>::const_iterator c = class_list.begin(); c != class_list.end(); ++c)
            filenames[*c] = files_in_directory(mode + "/" + *c, true);
        
        // set up the classifier based on the requested algo
        BOW *bowClassifier=0;
        bowClassifier = new BOW(class_list);
        
        // now train or test!
        if(mode == "train")
        {
            bowClassifier->train(filenames);
            system("./svm_multiclass_learn -c 0.1 bow-train-features bow-model ");
        }
        else if(mode == "test")
        {
            bowClassifier->test(filenames);
            system("./svm_multiclass_classify bow-test-features bow-model bow-predictions");
        }
        else
            throw std::string("unknown mode!");
    }
    catch(const string &err) {
        cerr << "Error: " << err << endl;
    }
}







