#pragma once

#include "../segm/tdef.h"

#include <vector>
#include <cstddef>

struct PixelPosition
{
    int row;
    int column;
};

class RegionList;

class ResultWrapper
{
public:
    ResultWrapper() : width(0), height(0) {}

    void init(size_t width, size_t height, const RegionList &regionList);

    size_t getNumRegions();
    
    // This method is used for visualization purposes only
    void setOutputImage(const std::vector<float> &imageData);

    // get the output image data
    std::vector<float> getOutputImage();

    std::vector<PixelPosition> getRegionBorder(size_t regionIndex);

private:
    std::vector<std::vector<PixelPosition> > borders; // To store the borders of the regions
    std::vector<float> outputImage; //To store the output image data

    size_t height;
    size_t width;
    
};

ResultWrapper meanShiftSegmentation(const unsigned char *data, int width, int height,
                                       float sigmaS, float sigmaR, int minRegion,
                                       Implementation implementation = SERIAL);


