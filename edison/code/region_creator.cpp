#include <chrono>
#include "region_creator.h"
#include "msImageProcessor.h"
#include <stdexcept>
#include <iostream>
#include <fstream>

void ResultWrapper::init(size_t width, size_t height, const RegionList &regionList)
{
    this->width = width;
    this->height = height;

    borders = std::vector<std::vector<PixelPosition>>(regionList.GetNumRegions());

    // print the number of borders
    std::cout << "Number of borders: " << borders.size() << std::endl;
    
    for (size_t i = 0; i < borders.size(); ++i)
    {
        borders[i].resize(regionList.GetRegionCount(i));
        for (size_t j = 0; j < borders[i].size(); ++j)
        {
            int index = regionList.GetRegionIndeces(i)[j];
            PixelPosition p;
            p.row = index / width;
            p.column = index % width;
            borders[i][j] = p;
        }
    }
}

void ResultWrapper::setOutputImage(const std::vector<float> &imageData)
{
    outputImage = imageData;
}

size_t ResultWrapper::getNumRegions()
{
    return borders.size();
}

std::vector<PixelPosition> ResultWrapper::getRegionBorder(size_t regionIndex)
{
    return borders[regionIndex];
}

// getOutputImage
std::vector<float> ResultWrapper::getOutputImage()
{
    return outputImage;
}

ResultWrapper meanShiftSegmentation(const unsigned char *data, int width, int height,
                                    float sigmaS, float sigmaR, int minRegion, Implementation implementation)
{
    

    std::vector<float> outputImage(width * height);
    msImageProcessor processor;
    processor.DefineImage(data, GRAYSCALE, height, width);


    // Time the execution of the filter
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_Filter = std::chrono::high_resolution_clock::now();
    processor.Filter(sigmaS, sigmaR, implementation);
    if (processor.ErrorStatus)
    {
        throw std::runtime_error("Filtering failed!");
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_Filter = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedFilter = end_time_Filter - start_time_Filter;
    std::cout << "Filtering time:\t" << elapsedFilter.count() << " s" << std::endl;

    // Time the execution of the region extraction
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_Fusion = std::chrono::high_resolution_clock::now();
    processor.FuseRegions(sigmaR, minRegion);
    if (processor.ErrorStatus)
    {
        throw std::runtime_error("Regions fusion failed!");
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_Fusion = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsedFusion = end_time_Fusion - start_time_Fusion;
    std::cout << "Region Fusing time:\t" << elapsedFusion.count() << " s" << std::endl;


    // create a ResultWrapper object to store the Boundaries coming from the image processor
    ResultWrapper regions;
    regions.init(width, height, *processor.GetBoundaries());

    processor.GetRawData(outputImage.data());

    // store the output image in a regions field
    regions.setOutputImage(outputImage);

    return regions;
}
