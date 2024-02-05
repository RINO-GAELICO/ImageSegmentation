#include "images.h"
#include "mean_shift.h"
#include <chrono>

#include <iostream>

using namespace images;

Image<unsigned char> loadImage(const std::string &filename);
Image<unsigned char> substitutePixelValues(Image<unsigned char> original, const std::vector<float> &outputImage, int height, int width);
Image<unsigned char> drawBorders(Image<unsigned char> original, int numOfRegions, ResultWrapper regions);

int main(int argc, char **argv)
{

    /*
    // STEPS FOR MEAN SHIFT SEGMENTATION IN MAIN:
    // 1.Set the parameters for the segmentation
    // 2.Load the image
    // 3.Run the segmentation
    // 4.Save the segmented image
    // 5.Save the image with the borders of the regions
    */

    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << "<input image> <area> <sigmaS> <sigmaR> <implementation>" << std::endl;
        return 1;
    }

    // Convert the char* to an int
    int sigmaS = atoi(argv[3]);
    int sigmaR = atoi(argv[4]);
    int minArea = atoi(argv[2]);

    // Convert the char* to a string
    std::string implementationStr = argv[5];

    // Convert the string to an enum value
    Implementation implementation;
    if (implementationStr == "SERIAL") {
        implementation = SERIAL;
    } else if (implementationStr == "OPEN_MP") {
        implementation = OPEN_MP;
    } else if (implementationStr == "CUDA") {
        implementation = CUDA;
    } else if (implementationStr == "MULTITHREADED_SPEEDUP") {
        implementation = MULTITHREADED_SPEEDUP;
    } else if (implementationStr == "GPU_SPEEDUP") {
        implementation = GPU_SPEEDUP;
    } else if (implementationStr == "AUTO_SPEEDUP") {
        implementation = AUTO_SPEEDUP;
    } else {
        std::cerr << "Unknown implementation: " << implementationStr << std::endl;
        return 1;
    }

    std::string inputName(argv[1]);
    // output name is the same as input name, but without the leading path
    size_t lastSlash = inputName.find_last_of('/');
    std::string outputName = inputName.substr(lastSlash + 1);

    // Load image
    Image<unsigned char> image = loadImage(inputName);
    if (image.isNull())
    {
        std::cerr << "Failed to load image: " << inputName << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << image.width << "x" << image.height << "x" << image.cn << std::endl;

    if (image.cn > 1)
    {
        // error image is not grayscale
        std::cerr << "Image is not grayscale" << std::endl;
        return 1;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    ResultWrapper regions = meanShiftSegmentation(image.ptr(), image.width, image.height, sigmaS, sigmaR, minArea, implementation);

    std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total segmentation time:\t" << elapsed.count() << " s" << std::endl;

    // Now, let's substitute the pixel values in the original image with the mean pixel value of the region
    // Use the outputImage from the segmentation and substitute the pixel values in the original image
    Image<unsigned char> original = image.copy();
    const std::vector<float> &outputImage = regions.getOutputImage(); // Assuming getOutputImage returns a const reference

    std::cout << "Coming from the segmentation: Image Size is " << outputImage.size() << std::endl;

    // we need to pass the processedImage, outputImage, and size of the image to the substitutePixelValues function
    Image<unsigned char> processedImage = substitutePixelValues(original, outputImage, image.height, image.width);

    std::cout << "Substituted Image Size: " << (processedImage.height * processedImage.width * processedImage.cn) << std::endl;
    processedImage.saveJPEG(outputName);
    std::cout << "Segmented image saved to " << outputName << std::endl;

    // Now, let's draw the borders of the regions on the image

    int numOfRegions = regions.getNumRegions();
    std::cout << "Number of regions found: " << numOfRegions << std::endl;
    Image<unsigned char> borderedImage = drawBorders(original, numOfRegions, regions);

    // Name of this output is the same as input but add "bordered" to the end
    outputName = outputName.substr(0, outputName.find_last_of('.')) + "_bordered.jpeg";
    borderedImage.savePNG(outputName);
    std::cout << "Bordered image saved to " << outputName << std::endl;

    return 0;
}

Image<unsigned char> loadImage(const std::string &filename)
{
    std::cout << "Loading image " << filename << "..." << std::endl;
    Image<unsigned char> image(filename);
    return image;
}

Image<unsigned char> substitutePixelValues(Image<unsigned char> original, const std::vector<float> &outputImage, int height, int width)
{
    Image<unsigned char> processedImage = original.copy();
    std::cout << "Substituting pixel values..." << std::endl;

    for (size_t j = 0; j < height; ++j)
    {
        for (size_t i = 0; i < width; ++i)
        {
            size_t index = j * width + i; // Assuming 1 channel (grayscale)
            unsigned char pixelValue = static_cast<unsigned char>(outputImage[index]);

            // Duplicate the pixel value across all three channels
            processedImage(j, i, 0) = pixelValue;
            processedImage(j, i, 1) = pixelValue;
            processedImage(j, i, 2) = pixelValue;
        }
    }

    return processedImage;
}

Image<unsigned char> drawBorders(Image<unsigned char> original, int numOfRegions, ResultWrapper regions)
{
    Image<unsigned char> borderedImage = original.copy();

    for (size_t i = 0; i < numOfRegions; ++i)
    {
        std::vector<PixelPosition> border = regions.getRegionBorder(i);
        for (size_t j = 0; j < border.size(); ++j)
        {
            for (size_t c = 0; c < borderedImage.cn; ++c)
            {
                // white border so it is visible
                borderedImage(border[j].row, border[j].column, c) = 255;
            }
        }
    }

    return borderedImage;
}
