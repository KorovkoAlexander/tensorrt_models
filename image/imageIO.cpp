//
// Created by alex on 12.09.19.
//

#include <imageIO.h>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

unsigned char* loadImageIO( const char* filename, int* width, int* height, int* channels )
{
    // validate parameters
    if( !filename || !width || !height )
    {
        printf("loadImageIO() - invalid parameter(s)\n");
        return nullptr;
    }

    // verify file path
    const std::string path = filename;

    if( path.length() == 0 )
    {
        printf("failed to find file '%s'\n", filename);
        return nullptr;
    }

    // load original image
    int imgWidth = 0;
    int imgHeight = 0;
    int imgChannels = 0;

    unsigned char* img = stbi_load(path.c_str(), &imgWidth, &imgHeight, &imgChannels, 0);

    if( !img )
    {
        printf( "failed to load '%s'\n", path.c_str());
        printf( "(error:  %s)\n", stbi_failure_reason());
        return nullptr;
    }

    // validate dimensions for sanity
    printf( "loaded '%s'  (%i x %i, %i channels)\n", filename, imgWidth, imgHeight, imgChannels);

    if( imgWidth < 0 || imgHeight < 0 || imgChannels < 0 || imgChannels > 4 )
    {
        printf( "'%s' has invalid dimensions\n", filename);
        return nullptr;
    }

    // if the user provided a desired size, resize the image if necessary
    const int resizeWidth  = *width;
    const int resizeHeight = *height;

    if( resizeWidth > 0 && resizeHeight > 0 && resizeWidth != imgWidth && resizeHeight != imgHeight )
    {
        unsigned char* img_org = img;

        printf( "resizing '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);

        // allocate memory for the resized image
        img = (unsigned char*)malloc(resizeWidth * resizeHeight * imgChannels * sizeof(unsigned char));

        if( !img )
        {
            printf( "failed to allocated memory to resize '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);
            free(img_org);
            return nullptr;
        }

        // resize the original image
        if( !stbir_resize_uint8(img_org, imgWidth, imgHeight, 0,
                                img, resizeWidth, resizeHeight, 0, imgChannels) )
        {
            printf( "failed to resize '%s' to %ix%i\n", filename, resizeWidth, resizeHeight);
            free(img_org);
            return nullptr;
        }

        // update resized dimensions
        imgWidth  = resizeWidth;
        imgHeight = resizeHeight;

        free(img_org);
    }

    *width = imgWidth;
    *height = imgHeight;
    *channels = imgChannels;

    return img;
}