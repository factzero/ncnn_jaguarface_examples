#include "featureextraction.h"


FeatureExtraction::FeatureExtraction(const std::string &bin_path, const std::string &param_path)
{
    face_ext_net_.load_param(param_path.data());
    face_ext_net_.load_model(bin_path.data());
}

FeatureExtraction::~FeatureExtraction()
{
    face_ext_net_.clear();
}

int FeatureExtraction::forward(ncnn::Mat &img, float *pfeature)
{
    ncnn::Mat out;

    ncnn::Extractor ex = face_ext_net_.create_extractor();
    ex.set_num_threads(4);
    ex.input("data", img);
    ex.extract("fc1", out);

    float *pfres = (float *)out.channel(0);
    float fl2 = 0.f;
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        fl2 += pfres[i] * pfres[i];
    }
    fl2 = std::sqrt(fl2);
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        pfeature[i] = pfres[i] / fl2;
    }

    return 0;
}
