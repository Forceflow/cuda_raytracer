#pragma once

#include <GL/glew.h>
#include <cstdio>

using namespace std;

void printglInfo(){
	printf("GL version: %s\n", glGetString(GL_VERSION));
	printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("Vendor: %s\n", glGetString(GL_VENDOR));
}