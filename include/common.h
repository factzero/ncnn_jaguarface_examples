#ifndef FACE_COMMON_H__
#define FACE_COMMON_H__

typedef struct Point 
{
    float x;
    float y;
} Point;

typedef struct FaceInfo
{
    float x1, y1;
    float x2, y2;
    float score;
    Point point[5];
} FaceInfo;

typedef struct bbox 
{
    float x1, y1;
    float x2, y2;
    float s;
    Point point[5];
} bbox;

typedef struct box 
{
    float cx;
    float cy;
    float sx;
    float sy;
} box;

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

#endif
