#include <iostream>
#include <algorithm>
#include <string>
#include <math.h>
#include "retinaFace.h"

RetinaFace::RetinaFace(const std::string &bin_path, const std::string &param_path):
    score_threshold_(0.6f), iou_threshold_(0.4f), mean_vals_{ 104.f, 117.f, 123.f },
    net_width_(320), net_height_(320)
{
    face_det_net_.load_param(param_path.data());
    face_det_net_.load_model(bin_path.data());

    //create_anchor(anchor_, net_width_, net_height_);
    create_anchor_retinaface(anchor_, net_width_, net_height_);
}

RetinaFace::~RetinaFace()
{
    face_det_net_.clear();
}

int RetinaFace::detect(ncnn::Mat &img, std::vector<bbox> &face_list)
{
    if (img.empty())
    {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }

#if 0
    //net_width_ = img.w;
    //net_height_ = img.h;
    int max_wh = std::max(img.w, img.h);
    float scale = 320.f / max_wh;
    net_width_ = int(scale*img.w);
    net_height_ = int(scale*img.h);
    //create_anchor(anchor_, net_width_, net_height_);
    create_anchor_retinaface(anchor_, net_width_, net_height_);
#endif

    ncnn::Mat out, out1, out2;
    ncnn::Mat img_det;
    ncnn::resize_bilinear(img, img_det, net_width_, net_height_);
    img_det.substract_mean_normalize(mean_vals_, 0);

    ncnn::Extractor ex = face_det_net_.create_extractor();
    ex.set_num_threads(4);
    ex.input(0, img_det);
    ex.extract("output0", out); // loc
    ex.extract("586", out1);    // class   
    ex.extract("585", out2);    // landmark

    generate_bboxes(face_list, anchor_, (float *)out.channel(0), (float *)out1.channel(0), (float *)out2.channel(0),
        score_threshold_, img.w, img.h);
    nms(face_list, iou_threshold_);

    return 0;
}

void RetinaFace::create_anchor(std::vector<box> &anchor, int w, int h)
{
    anchor.clear();
    std::vector<std::vector<int> > feature_map(4);
    float steps[] = { 8.f, 16.f, 32.f, 64.f };
    for (int i = 0; i < feature_map.size(); i++)
    {
        feature_map[i].push_back((int)ceil(h / steps[i]));
        feature_map[i].push_back((int)ceil(w / steps[i]));
    }

    std::vector<std::vector<int> > min_sizes = {
            {10,  16,  24},
            {32,  48},
            {64,  96},
            {128, 192, 256}};
    for (int k = 0; k < feature_map.size(); k++)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; i++)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l] * 1.0f / w;
                    float s_ky = min_size[l] * 1.0f / h;
                    float cx = (j + 0.5f) * steps[k] / w;
                    float cy = (i + 0.5f) * steps[k] / h;
                    box axil = { cx, cy, s_kx, s_ky };
                    anchor.push_back(axil);
                }
            }
        }
    }

    return;
}

void RetinaFace::create_anchor_retinaface(std::vector<box> &anchor, int w, int h)
{
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3);
    float steps[] = { 8.f, 16.f, 32.f };
    for (int i = 0; i < feature_map.size(); i++)
    {
        feature_map[i].push_back((int)ceil(h / steps[i]));
        feature_map[i].push_back((int)ceil(w / steps[i]));
    }

    std::vector<std::vector<int> > min_sizes = {
            { 10,  20},
            { 32,  64},
            {128, 256} };
    for (int k = 0; k < feature_map.size(); k++)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; i++)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l] * 1.0f / w;
                    float s_ky = min_size[l] * 1.0f / h;
                    float cx = (j + 0.5f) * steps[k] / w;
                    float cy = (i + 0.5f) * steps[k] / h;
                    box axil = { cx, cy, s_kx, s_ky };
                    anchor.push_back(axil);
                }
            }
        }
    }

    return;
}

void RetinaFace::generate_bboxes(std::vector<bbox> &bboxes, const std::vector<box> &anchor,
    float *pfloc, float *pfcls, float *pflandmark, float score_threshold, int w, int h)
{
    float *pfprobtmp = pfloc;
    float *pfclstmp = pfcls;
    float *pflandmarktmp = pflandmark;
    bboxes.clear();
    for (int i = 0; i < anchor.size(); i++)
    {
        if (*(pfclstmp+1) > score_threshold)
        {
            box tmp = anchor[i];
            box tmp1;            
            // loc and conf
            tmp1.cx = tmp.cx + *pfprobtmp * 0.1f * tmp.sx;
            tmp1.cy = tmp.cy + *(pfprobtmp+1) * 0.1f * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(pfprobtmp + 2) * 0.2f);
            tmp1.sy = tmp.sy * exp(*(pfprobtmp + 3) * 0.2f);

            bbox result;
            result.x1 = (tmp1.cx - tmp1.sx / 2) * w;
            result.x1 = std::max(result.x1, 0.f);
            result.y1 = (tmp1.cy - tmp1.sy / 2) * h;
            result.y1 = std::max(result.y1, 0.f);
            result.x2 = (tmp1.cx + tmp1.sx / 2) * w;
            result.x2 = std::min(result.x2, (float)w);
            result.y2 = (tmp1.cy + tmp1.sy / 2) * h;
            result.y2 = std::min(result.y2, (float)h);
            result.s = *(pfclstmp + 1);

            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.point[j].x = (tmp.cx + *(pflandmarktmp + (j << 1)) * 0.1f * tmp.sx) * w;
                result.point[j].y = (tmp.cy + *(pflandmarktmp + (j << 1) + 1) * 0.1f * tmp.sy) * h;
            }

            bboxes.push_back(result);
        }

        pfprobtmp += 4;
        pfclstmp += 2;
        pflandmarktmp += 10;
    }

    return;
}

void RetinaFace::nms(std::vector<bbox> &bboxes, float iou_threshold)
{
    std::sort(bboxes.begin(), bboxes.end(), [](const bbox &a, const bbox &b) { return a.s > b.s; });
    std::vector<float> areas(bboxes.size());
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        areas[i] = (bboxes[i].x2 - bboxes[i].x1 + 1) * (bboxes[i].y2 - bboxes[i].y1 + 1);
    }

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        for (size_t j = i+1; j < bboxes.size();)
        {
            float xx1 = std::max(bboxes[i].x1, bboxes[j].x1);
            float yy1 = std::max(bboxes[i].y1, bboxes[j].y1);
            float xx2 = std::min(bboxes[i].x2, bboxes[j].x2);
            float yy2 = std::min(bboxes[i].y2, bboxes[j].y2);
            float w = std::max(0.f, xx2 - xx1 + 1);
            float h = std::max(0.f, yy2 - yy1 + 1);
            float inter = w * h;
            float score = inter / (areas[i] + areas[j] - inter);
            if (score > iou_threshold)
            {
                bboxes.erase(bboxes.begin() + j);
                areas.erase(areas.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }

    return;
}
