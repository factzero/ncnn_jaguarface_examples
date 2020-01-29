#ifndef FEATURE_EXTRACTION_H__
#define FEATURE_EXTRACTION_H__

#include "ncnn/net.h"

#define FEATURE_LEN  128

class FeatureExtraction
{
public:
    FeatureExtraction(const std::string &bin_path, const std::string &param_path);
    ~FeatureExtraction();
    int forward(ncnn::Mat &img, float *pfeature);

private:
    ncnn::Net face_ext_net_;
};

#endif
