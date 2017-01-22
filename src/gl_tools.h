#pragma once

#include <GL/glew.h>
#include <cstdio>

using namespace std;

////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return bool if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
//! @note This function should be used via the CHECK_ERROR_GL() macro
////////////////////////////////////////////////////////////////////////////
inline bool
sdkCheckErrorGL(const char *file, const int line)
{
	bool ret_val = true;

	// check for error
	GLenum gl_error = glGetError();

	if (gl_error != GL_NO_ERROR)
	{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		char tmpStr[512];
		// NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
		// when the user double clicks on the error line in the Output pane. Like any compile error.
		sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, gluErrorString(gl_error));
		fprintf(stderr, "%s", tmpStr);
#endif
		fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
		fprintf(stderr, "%s\n", gluErrorString(gl_error));
		ret_val = false;
	}

	return ret_val;
}

#define SDK_CHECK_ERROR_GL() sdkCheckErrorGL( __FILE__, __LINE__)

//#define SDK_CHECK_ERROR_GL()                                              \
//    if( false == sdkCheckErrorGL( __FILE__, __LINE__)) {                  \
//        exit(EXIT_FAILURE);                                               \
//		    }

void printGlewInfo(){
	printf("GLEW: Glew version: %s \n", glewGetString(GLEW_VERSION));
}

void printGLInfo(){
	printf("OpenGL: GL version: %s \n", glGetString(GL_VERSION));
	printf("OpenGL: GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("OpenGL: Vendor: %s\n", glGetString(GL_VENDOR));
}