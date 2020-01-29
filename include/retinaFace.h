#ifndef FACE_RETINAFACE_H__
#define FACE_RETINAFACE_H__

#include "ncnn/net.h"
#include "common.h"


class RetinaFace
{
public:
    RetinaFace(const std::string &bin_path, const std::string &param_path);
    ~RetinaFace();
    int detect(ncnn::Mat &img, std::vector<bbox> &face_list);

private:
    void create_anchor(std::vector<box> &anchor, int w, int h);
    void create_anchor_retinaface(std::vector<box> &anchor, int w, int h);
    void generate_bboxes(std::vector<bbox> &bboxes, const std::vector<box> &anchor,
        float *pfloc, float *pfcls, float *pflandmark, float score_threshold, int w, int h);
    void nms(std::vector<bbox> &bboxes, float iou_threshold);

private:
    ncnn::Net face_det_net_;

    float score_threshold_;
    float iou_threshold_;
    float mean_vals_[3];
    int net_width_;
    int net_height_;

    std::vector<box> anchor_;
};

#endif
