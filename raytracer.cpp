#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <fstream>
#include <float.h>
#include <random>
#include <chrono>
#include <SDL.h>
#include <direct.h>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMEMTATION
#include "stb_image/stb_image.h"
#endif	/* STB_IMAGE_IMPLEMENTATION */

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#endif

#include "hitables/sphere.h"
#include "hitables/hitable_list.h"
#include "util/camera.h"
#include "util/renderer.h"
#include "util/window.h"
#include "util/common.h"
#include "util/scene.h"
#include "materials/material.h"

#include <sys/types.h>
#include <sys/stat.h>

#ifdef CUDA_ENABLED
void initializeWorldCuda(bool showWindow, bool writeImagePPM, bool writeImagePNG, hitable*** list, hitable** world, Window** w, Image** image, camera** cam, Renderer** render);
void destroyWorldCuda(bool showWindow, hitable** list, hitable* world, Window* w, Image* image, camera* cam, Renderer* render);
#else
void initializeWorld(bool showWindow, bool writeImagePPM, bool writeImagePNG, hitable** world, Window** w, Image** image, Camera** cam, Renderer** render)
{
    *image = new Image(showWindow, writeImagePPM || writeImagePNG, nx, ny, tx, ty);
    vec3 lookFrom(13.0f, 2.0f, 3.0f);
    vec3 lookAt(0.0f, 0.0f, 0.0f);
    *cam = new Camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f), 20.0f, float(nx) / float(ny), distToFocus);
    *render = new Renderer(showWindow, writeImagePPM, writeImagePNG);
    *world = simpleScene();

    if (showWindow)
        *w = new Window(*cam, *render, nx, ny, thetaInit, phiInit, zoomScale, stepScale);
}
#endif	// CUDA_ENABLED

void invokeRenderer(hitable* world, Window* w, Image* image, camera* cam, Renderer* render, bool showWindow, bool writeImagePPM, bool writeImagePNG, bool writeEveryImageToFile)
{
    std::ofstream ppmImageStream;

    if (writeImagePPM)
    {
        ppmImageStream.open("test.ppm");
        if (ppmImageStream.is_open())
            ppmImageStream << "P3\n" << nx << " " << ny << "\n255\n";
        else
            std::cout << "Unable to open file" << std::endl;
    }

    if (writeEveryImageToFile)
    {
        // make the folder
        std::string path = "./" + folderName;
        //mode_t mode = 0733;
        int error = 0;
#if defined(_WIN64)
        error = _mkdir(path.c_str());
#else
        error = mkdir(path.c_str(), mode);
#endif
        if (error != 0)
            std::cerr << "Couldn't create output folder." << std::endl;
    }

    int numberOfIterations = ns;
    if (showWindow)
    {
        int j = 1;
        for (int i = 0; i < numberOfIterations; ++i, ++j)
        {
            w->updateImage(showWindow, writeImagePPM, writeImagePNG, ppmImageStream, w, cam, world, image, i + 1, image->fileOutputImage);
            w->pollEvents(image, image->fileOutputImage);
            
            if (writeEveryImageToFile && (j % sampleNrToWrite == 0))
            {
                w->moveCamera(image, image->fileOutputImage);
                j = 0;
            }
            if (w->refresh)
            {
                std::string currentFileName(folderName + "/" + fileName);
                currentFileName += formatNumber(imageNr);
                imageNr++;
                currentFileName += ".png";
                // write png
                stbi_write_png(currentFileName.c_str(), nx, ny, 3, image->fileOutputImage, nx * 3);
                image->resetImage();
                i = -1;
                w->refresh = false;
            }
            if (w->quit)
                break;
        }
        std::cout << "Done." << std::endl;

        // we write the files after the windows is closed
        if (writeImagePPM)
        {
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    ppmImageStream << int(image->fileOutputImage[(j * nx + i) * 3]) << " " << int(image->fileOutputImage[(j * nx + i) * 3 + 1]) << " " << int(image->fileOutputImage[(j * nx + i) * 3 + 2]) << "\n";
            ppmImageStream.close();
        }

        if (writeImagePNG)
        {
            // write png
            stbi_write_png("test.png", nx, ny, 3, image->fileOutputImage, nx * 3);
        }
    }
    else
    {
        for (int i = 0; i < numberOfIterations; ++i)
            render->trace_rays(nullptr, cam, world, image, i + 1, image->fileOutputImage);
        std::cout << "Done." << std::endl;
    }
}

void raytrace(bool showWindow, bool writeImagePPM, bool writeImagePNG, bool writeEveryImageToFile)
{
    Window* w;
    Image* image;
    camera* cam;
    Renderer* render;
    hitable* world;
    hitable** list;

#ifdef CUDA_ENABLED
    initializeWorldCuda(showWindow, writeImagePPM, writeImagePNG, &list, &world, &w, &image, &cam, &render);
    invokeRenderer(world, w, image, cam, render, showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile);
    destroyWorldCuda(showWindow, list, world, w, image, cam, render);
#else
    initializeWorld(showWindow, writeImagePPM, writeImagePNG, &world, &w, &image, &cam, &render);
    invokeRenderer(world, w, image, cam, render, showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile);

    delete image;
    delete cam;
    delete render;
    delete world;
    if (showWindow)
        delete w;
#endif  // CUDA_ENABLED
}

int main(int argc, char** argv)
{
    bool writeImagePPM = true;
    bool writeImagePNG = true;
    bool showWindow = true;
    bool runBenchmark = false;
    bool writeEveryImageToFile = true;

    if (runBenchmark)
    {
        std::ofstream benchmarkStream;

        for (int i = 0; i < benchmarkCount; ++i)
        {
            benchmarkStream.open("benchmark/benchmarkResultCUDA.txt");
            // Record start time
            auto start = std::chrono::high_resolution_clock::now();

            // Invoke renderer
            raytrace(false, false, false, false);

            // Record end time
            auto finish = std::chrono::high_resolution_clock::now();

            // Compute elapsed time
            std::chrono::duration<double> elapsed = finish - start;

            // Write results to file
            benchmarkStream << ns << " " << elapsed.count() << "s\n";

            benchmarkStream.close();
        }
    }
    else
    {
        // Invoke renderer
        raytrace(showWindow, writeImagePPM, writeImagePNG, writeEveryImageToFile);
    }

    return 0;
}