#pragma once
#include "stb_image.h"
#include <GL/glew.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
GLuint loadTextureImage(const char* imagepath);
GLuint loadCubemap(vector <std::string> faces);