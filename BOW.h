
#include<map>
#include<vector>
#include<string>
#include<iostream>
#include<ctime>
#include<cmath>
#include <fstream>
#include <Sift.h>

using namespace std;

bool compareNoCase (string first, string second)
{
    int i=0;
    while ((i < first.length()) && (i < second.length()))
    {
        if (tolower (first[i]) < tolower (second[i])) return true;
        else if (tolower (first[i]) > tolower (second[i])) return false;
        i++;
    }
    
    if (first.length() < second.length()) return true;
    else return false;
}

class BOW
{
public:
    BOW(const vector<string> &_class_list) : class_list(_class_list) {}
    
    // Run training on a given dataset.
    virtual void train(const Dataset &filenames)
    {
        std::vector<std::vector<float> > training_descriptors;
        std::vector<int> descriptor_counts;
        int num_images = 0;
        for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
        {
            for(int i=0; i<c_iter->second.size(); i++)
            {
                
                CImg<float> img(c_iter->second[i].c_str());
                img = img.blur(2.0,true,true);
                for(int size=0; size<3; size++)
                {
                    img = img.resize_halfXY();
                    std::vector<SiftDescriptor> img_sift_descriptors = Sift::compute_sift(img);
                
                    int num_descriptors = 0;
                    num_images++;
                    for(int j=0; j< (int)img_sift_descriptors.size(); j++)
                    {
                        if((int) (img_sift_descriptors[j].descriptor.size())==128)
                        {
                            training_descriptors.push_back(img_sift_descriptors[j].descriptor);
                            num_descriptors++;
                        }
                        else
                        {
                            cout<<endl<<"\nError Computing Sift";
                        }
                    }
                    descriptor_counts.push_back(num_descriptors);
                }
            }
        }
        
        cout<<"Training "<<(int)training_descriptors.size()<<" descriptors"<<endl;
        
        Mat descriptors((int)(training_descriptors.size()), 128, CV_32F);
        Mat labels;
        for(int i=0; i<training_descriptors.size(); i++)
        {
            for(int j=0; j<128; j++)
            {
                descriptors.at<float>(i,j) = training_descriptors[i][j];
            }
        }
        
        int k = 250;
        
        cout<<"\n Clustering ..\n"<<descriptors.rows<<" descriptors";
        
        kmeans(descriptors,k,labels,TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 100,0.1), 5, KMEANS_PP_CENTERS, centers);
        
        cout<<endl<<"labels shape "<<labels.rows<<" x "<<labels.cols;
        
        FileStorage f("word_cluster_centers.xml", FileStorage::WRITE);
        f<<"centers"<<centers;
        
        Dataset::const_iterator c_iter=filenames.begin();
        ofstream fout("bow-train-features");
        cout<<"\n Building SVM data";
        int s=0,cl=1;
        
        int num_imgs=0;
        for(int j=0;j<(int)(descriptor_counts.size());j++,num_imgs++)
        {
            vector<float> hist;
            for(int i=0;i<k;i++)
            {
                hist.push_back(0);
            }
            cout<<"\n Building histogram for "<<j+1<<"th image ";
            if (num_imgs == (int)c_iter->second.size())
            {
                cl++;
                c_iter++;
                num_imgs=0;
            }
            fout<<cl;
            for(int i=s; i < s+descriptor_counts[j] && i< labels.rows; i++)
            {
                if(labels.at<int>(i)<k)
                {
                    hist[labels.at<int>(i)]++;
                }
                else
                {
                    cout<<endl<<labels.at<int>(i);
                }
            }
            
            for(int i=0;i<k;i++)
            {
                fout<<'\t'<<i+1<<':'<<(hist[i]/descriptor_counts[j]);
            }
            
            fout<<'\n';
            s+=descriptor_counts[j];
        }
        
        fout.close();
    }
    
    int get_cluster_index(vector<float> descriptor)
    {
        
        int cluster=0;
        float distance=100000000.0;
        for(int i=0;i<centers.rows;i++)
        {
            float desc_dist = 0.0;
            for(int j=0;j<128;j++)
            {
                desc_dist += (centers.at<float>(i,j) - descriptor[j])*(centers.at<float>(i,j) - descriptor[j]);
            }
            
            if(desc_dist<distance)
            {
                distance  = desc_dist;
                cluster = i;
            }
        }
        return cluster;
    }
    
    // Classify a single image.
    virtual string classify(const string &filename)
    {
        return 0;
    }
    
    // Load in a trained model.
    virtual void load_model()
    {
        FileStorage f("word_cluster_centers.xml", FileStorage::READ);
        f["centers"]>>centers;
    }
    
    // Loop through all test images, hiding correct labels and checking if we get them right.
    virtual void test(const Dataset &filenames)
    {
        if(centers.rows == 0)
        {
            load_model();
        }
        
        ofstream fout("bow-test-features");
        int k=1;
        
        for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter, ++k)
        {
            cout<<endl<<" Extracting descriptors for "<<c_iter->first;
            for(int i=0; i<c_iter->second.size(); i++)
            {
                CImg<float> img(c_iter->second[i].c_str());
                fout<<k;
                vector<float> hist;
                for(int j=0;j<centers.rows;j++)
                {
                    hist.push_back(0);
                }
                
                vector<SiftDescriptor> img_sift_descriptors = Sift::compute_sift(img.blur(2.0,true,true).resize_halfXY().resize_halfXY().resize_halfXY() );
                
                for(int j=0;j<img_sift_descriptors.size();j++)
                {
                    if(img_sift_descriptors[j].descriptor.size()==128)
                    {
                        int cl = get_cluster_index(img_sift_descriptors[j].descriptor);
                        hist[cl]++;
                    }
                }
                for(int j=0;j<hist.size();j++)
                {
                    fout<<'\t'<<j+1<<":"<<(hist[j]/img_sift_descriptors.size());
                }
                fout<<'\n';
            }
        }
        fout.close();
        /*
        // now score!
        map< string, map< string, double > > confusion;
        int correct=0, total=0;
        for(map<string, vector<string> >::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
            for(vector<string>::const_iterator f_iter = c_iter->second.begin(); f_iter != c_iter->second.end(); ++f_iter, ++total)
                confusion[c_iter->first][ predictions[c_iter->first][*f_iter] ]++;
        
        cout << "Confusion matrix:" << endl << setw(20) << " " << " ";
        for(int j=0; j<class_list.size(); j++)
            cout << setw(2) << class_list[j].substr(0, 2) << " ";
        
        for(int i=0; i<class_list.size(); i++)
        {
            cout << endl << setw(20) << class_list[i] << " ";
            for(int j=0; j<class_list.size(); j++)
                cout << setw(2) << confusion[ class_list[i] ][ class_list[j] ] << (j==i?".":" ");
            
            correct += confusion[ class_list[i] ][ class_list[i] ];
        }
        
        cout << endl << "Classifier accuracy: " << correct << " of " << total << " = " << setw(5) << setprecision(2) << correct/double(total)*100 << "%";
        cout << "  (versus random guessing accuracy of " << setw(5) << setprecision(2) << 1.0/class_list.size()*100 << "%)" << endl;*/
    }
    
protected:
    vector<string> class_list;
    Mat centers;
};
