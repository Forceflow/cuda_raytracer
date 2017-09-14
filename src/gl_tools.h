#pragma once

#include <GL/glew.h>
#include <cstdio>
#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

using namespace std;

void printGlewInfo(){
	printf("GLEW: Glew version: %s \n", glewGetString(GLEW_VERSION));
}

void printGLInfo(){
	printf("OpenGL: GL version: %s \n", glGetString(GL_VERSION));
	printf("OpenGL: GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("OpenGL: Vendor: %s\n", glGetString(GL_VENDOR));
}

GLuint loadTextureFromFile(std::string filepath) {
	GLuint texture0;
	glGenTextures(1, &texture0); // Load simple OpenGL texture
	glBindTexture(GL_TEXTURE_2D, texture0); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
											// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// load image, create texture and generate mipmaps
	int width, height, nrChannels;
	printf("Loading texture from %s ... ", filepath.c_str());
	unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 0);
	if (data) {
		printf("OK - Texture ID: (%i) \n", texture0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else { printf("Error: No texture found. \n"); }
	stbi_image_free(data);
	return texture0;
}