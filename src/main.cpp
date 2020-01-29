#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "retinaFace.h"
#include "featureextraction.h"

void alignmentFace(cv::Mat &input, bbox &face_bbox, int out_w, int out_h, cv::Mat &output);

class TestFace
{
public:
    TestFace(const std::string &det_bin, const std::string &det_param, const std::string &feature_bin, const std::string &feature_param);
    ~TestFace();
    int getFeature(cv::Mat &img, float *pfeature);

private:
    std::shared_ptr<RetinaFace> retinaface_; 
    std::shared_ptr<FeatureExtraction> face_feature_extraction_;
};

TestFace::TestFace(const std::string &det_bin, const std::string &det_param, const std::string &feature_bin, const std::string &feature_param)
{
    retinaface_ = std::make_shared<RetinaFace>(det_bin, det_param);
    face_feature_extraction_ = std::make_shared<FeatureExtraction>(feature_bin, feature_param);
}

TestFace::~TestFace()
{
    
}

int TestFace::getFeature(cv::Mat &img, float *pfeature)
{
    std::vector<bbox> boxes;
    cv::Mat roi_face;
    ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
    retinaface_->detect(inmat, boxes);
    if (0 == boxes.size())
    {
        std::cout << "image has no face !!! " << std::endl;
        return -1;
    }

    for (int j = 0; j < boxes.size(); ++j)
    {
        alignmentFace(img, boxes[j], 112, 112, roi_face);
        cv::Rect rect((int)boxes[j].x1, (int)boxes[j].y1, (int)boxes[j].x2 - (int)boxes[j].x1, (int)boxes[j].y2 - (int)boxes[j].y1);
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
        char test[80];
        sprintf_s(test, "%f", boxes[j].s);

        cv::putText(img, test, cv::Size((int)boxes[j].x1, (int)boxes[j].y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
        cv::circle(img, cv::Point((int)boxes[j].point[0].x, (int)boxes[j].point[0].y), 1, cv::Scalar(0, 0, 225), 4);
        cv::circle(img, cv::Point((int)boxes[j].point[1].x, (int)boxes[j].point[1].y), 1, cv::Scalar(0, 255, 225), 4);
        cv::circle(img, cv::Point((int)boxes[j].point[2].x, (int)boxes[j].point[2].y), 1, cv::Scalar(255, 0, 225), 4);
        cv::circle(img, cv::Point((int)boxes[j].point[3].x, (int)boxes[j].point[3].y), 1, cv::Scalar(0, 255, 0), 4);
        cv::circle(img, cv::Point((int)boxes[j].point[4].x, (int)boxes[j].point[4].y), 1, cv::Scalar(255, 0, 0), 4);
    }

    cv::imshow("retinaFace", img);
    cv::waitKey();

    ncnn::Mat align_face = ncnn::Mat::from_pixels(roi_face.data, ncnn::Mat::PIXEL_BGR2RGB, roi_face.cols, roi_face.rows);
    face_feature_extraction_->forward(align_face, pfeature);

    return 0;
}

void test_feature_cmp()
{
    std::string det_bin = "../models/retinaface/face.bin";
    std::string det_param = "../models/retinaface/face.param";
    std::string feature_bin = "../models/facefeature/ncnn.bin";
    std::string feature_param = "../models/facefeature/ncnn.param";
    std::shared_ptr<TestFace> test_face_feature = std::make_shared<TestFace>(det_bin, det_param, feature_bin, feature_param);

    std::string image_file1 = "../data/jf1.jpg";
    std::string image_file2 = "../data/jf.jpg";
    cv::Mat frame1 = cv::imread(image_file1);
    cv::Mat frame2 = cv::imread(image_file2);

    float affeature1[FEATURE_LEN] = { 0 };
    float affeature2[FEATURE_LEN] = { 0 };

    test_face_feature->getFeature(frame1, affeature1);
    test_face_feature->getFeature(frame2, affeature2);

    float fcos = 0.f;
    for (int i = 0; i < FEATURE_LEN; i++)
    {
        fcos += affeature1[i] * affeature2[i];
    }

    std::cout << "fcos =  " << fcos << std::endl;
}

void alignmentFace(cv::Mat &input, bbox &face_bbox, int out_w, int out_h, cv::Mat &output)
{
    const int EDGE = 64;
    int w = input.cols;
    int h = input.rows;
    bbox border_bbox;
    border_bbox.x1 = std::max(face_bbox.x1 - EDGE, 0.f);
    border_bbox.y1 = std::max(face_bbox.y1 - EDGE, 0.f);
    border_bbox.x2 = std::min(face_bbox.x2 + EDGE, (float)w);
    border_bbox.y2 = std::min(face_bbox.y2 + EDGE, (float)h);   

    cv::Mat roi_image, resize_image;
    cv::Rect roi((int)border_bbox.x1, (int)border_bbox.y1, (int)border_bbox.x2 - (int)border_bbox.x1, (int)border_bbox.y2 - (int)border_bbox.y1);
    input(roi).copyTo(roi_image);
    cv::resize(roi_image, resize_image, cv::Size(out_w, out_h));
    float scale_w = (float)out_w / roi_image.cols;
    float scale_h = (float)out_h / roi_image.rows;

    cv::Point2f dst_points[5] = {
        cv::Point2f(38.2946f, 51.6963f),
        cv::Point2f(73.5318f, 51.5014f),
        cv::Point2f(56.0252f, 71.7366f),
        cv::Point2f(41.5493f, 92.3655f),
        cv::Point2f(70.7299f, 92.2041f)
    };
    cv::Point2f src_points[5] = {
        cv::Point2f((face_bbox.point[0].x - border_bbox.x1)*scale_w, (face_bbox.point[0].y - border_bbox.y1)*scale_h),
        cv::Point2f((face_bbox.point[1].x - border_bbox.x1)*scale_w, (face_bbox.point[1].y - border_bbox.y1)*scale_h),
        cv::Point2f((face_bbox.point[2].x - border_bbox.x1)*scale_w, (face_bbox.point[2].y - border_bbox.y1)*scale_h),
        cv::Point2f((face_bbox.point[3].x - border_bbox.x1)*scale_w, (face_bbox.point[3].y - border_bbox.y1)*scale_h),
        cv::Point2f((face_bbox.point[4].x - border_bbox.x1)*scale_w, (face_bbox.point[4].y - border_bbox.y1)*scale_h)
    };

    std::vector<cv::Point2f>  p1s, p2s;
    for (int i = 0; i < 5; i++)
    {
        p1s.push_back(src_points[i]);
        p2s.push_back(dst_points[i]);
    }
    cv::Mat t = cv::estimateRigidTransform(p1s, p2s, true);
    cv::warpAffine(resize_image, output, t, resize_image.size());
    //output = resize_image;

    /*cv::circle(output, cv::Point((int)src_points[0].x, (int)src_points[0].y), 1, cv::Scalar(0, 0, 225), 4);
    cv::circle(output, cv::Point((int)src_points[1].x, (int)src_points[1].y), 1, cv::Scalar(0, 255, 225), 4);
    cv::circle(output, cv::Point((int)src_points[2].x, (int)src_points[2].y), 1, cv::Scalar(255, 0, 225), 4);
    cv::circle(output, cv::Point((int)src_points[3].x, (int)src_points[3].y), 1, cv::Scalar(0, 255, 0), 4);
    cv::circle(output, cv::Point((int)src_points[4].x, (int)src_points[4].y), 1, cv::Scalar(255, 0, 0), 4);*/
    cv::circle(output, cv::Point((int)dst_points[0].x, (int)dst_points[0].y), 1, cv::Scalar(0, 0, 225), 4);
    cv::circle(output, cv::Point((int)dst_points[1].x, (int)dst_points[1].y), 1, cv::Scalar(0, 255, 225), 4);
    cv::circle(output, cv::Point((int)dst_points[2].x, (int)dst_points[2].y), 1, cv::Scalar(255, 0, 225), 4);
    cv::circle(output, cv::Point((int)dst_points[3].x, (int)dst_points[3].y), 1, cv::Scalar(0, 255, 0), 4);
    cv::circle(output, cv::Point((int)dst_points[4].x, (int)dst_points[4].y), 1, cv::Scalar(255, 0, 0), 4);

    return;
}

void test_retinaface()
{
    std::string bin_path = "../models/retinaface/face.bin";
    std::string param_path = "../models/retinaface/face.param";
    std::string feature_bin_path = "../models/FaceFeature/ncnn.bin";
    std::string feature_param_path = "../models/FaceFeature/ncnn.param";
    RetinaFace retinaface(bin_path, param_path);
    FeatureExtraction face_feature_extraction(feature_bin_path, feature_param_path);
    float affeature[FEATURE_LEN] = { 0 };


    // for (int i = 3; i < argc; i++) 
    {
        std::string image_file = "../data/222.jpg";
        std::cout << "Processing " << image_file << std::endl;

        cv::Mat frame = cv::imread(image_file);
        ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

        std::vector<bbox> boxes;
        retinaface.detect(inmat, boxes);
        
        cv::Mat roi_face;
        for (int j = 0; j < boxes.size(); ++j)
        {
            alignmentFace(frame, boxes[j], 112, 112, roi_face);
            cv::Rect rect((int)boxes[j].x1, (int)boxes[j].y1, (int)boxes[j].x2 - (int)boxes[j].x1, (int)boxes[j].y2 - (int)boxes[j].y1);
            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
            char test[80];
            sprintf_s(test, "%f", boxes[j].s);

            cv::putText(frame, test, cv::Size((int)boxes[j].x1, (int)boxes[j].y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
            cv::circle(frame, cv::Point((int)boxes[j].point[0].x, (int)boxes[j].point[0].y), 1, cv::Scalar(0, 0, 225), 4);
            cv::circle(frame, cv::Point((int)boxes[j].point[1].x, (int)boxes[j].point[1].y), 1, cv::Scalar(0, 255, 225), 4);
            cv::circle(frame, cv::Point((int)boxes[j].point[2].x, (int)boxes[j].point[2].y), 1, cv::Scalar(255, 0, 225), 4);
            cv::circle(frame, cv::Point((int)boxes[j].point[3].x, (int)boxes[j].point[3].y), 1, cv::Scalar(0, 255, 0), 4);
            cv::circle(frame, cv::Point((int)boxes[j].point[4].x, (int)boxes[j].point[4].y), 1, cv::Scalar(255, 0, 0), 4);
        }

        ncnn::Mat align_face = ncnn::Mat::from_pixels(roi_face.data, ncnn::Mat::PIXEL_BGR2RGB, roi_face.cols, roi_face.rows);
        face_feature_extraction.forward(align_face, affeature);

        float fsum = 0.f;
        for (int i = 0; i < FEATURE_LEN; i++)
        {
            fsum += affeature[i] * affeature[i];
        }
        std::cout << "fsum =  " << fsum << std::endl;

        cv::imshow("retinaFace", frame);
        cv::imshow("roiFace", roi_face);
        cv::waitKey();
        cv::imwrite("../data/result_retina.jpg", frame);
    }

    return;
}

int main(int argc, char **argv) 
{
    //test_retinaface();
    test_feature_cmp();

    return 0;
}
