#pragma once

#include <GL/glew.h>
#include <cstdio>

using namespace std;

void printglewInfo(){
	printf("GLEW: Glew version: %s \n", glewGetString(GLEW_VERSION));
}

void printglInfo(){
	printf("OpenGL: GL version: %s \n", glGetString(GL_VERSION));
	printf("OpenGL: GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("OpenGL: Vendor: %s\n", glGetString(GL_VENDOR));
}