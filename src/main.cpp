#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

#include "cuda_error_check.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;

// GLFW
GLFWwindow* window;
int width = 512;
int height = 512;

void display(void){
	glClear(GL_COLOR_BUFFER_BIT);
	glfwSwapBuffers(window);
	// call to display image here
}

// Keyboard
void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods){
	switch (key) {
	}
}

void parseProgramParameters(int argc, char* argv[]){
	if (argc<2){ // not enough arguments
		exit(0);
	}
	for (int i = 1; i < argc; i++) {
	}
}

bool initGL(){
	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported(
		"GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		))
	{
		printf("ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
#ifndef USE_TEXTURE_RGBA8UI
	glClearColor(0.5, 0.5, 0.5, 1.0); // specify what color used when clearing
#else
	glClearColorIuiEXT(128, 128, 128, 255);
#endif
	glDisable(GL_DEPTH_TEST);

	glViewport(0, 0, width, height); // viewport for x,y to normalized device coordinates transformation

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)width / (GLfloat)height, 0.1f, 10.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	/*glEnable(GL_LIGHT0);
	float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);*/

	//SDK_CHECK_ERROR_GL();
	return true;
}

bool initGLFW(){
	if (!glfwInit()) exit(EXIT_FAILURE);
	window = glfwCreateWindow(width, height, "Voxel Ray Caster", NULL, NULL);
	if (!window){ glfwTerminate(); exit(EXIT_FAILURE); }
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glfwSetKeyCallback(window, keyboardfunc);
	return true;
}

int main(int argc, char *argv[]) {
	fprintf(stdout, "\n## PROGRAM PARAMETERS \n");
	checkCudaRequirements();

	initGLFW();
	initGL();

	while (!glfwWindowShouldClose(window))
	{
		display();
		glfwWaitEvents();
	}
	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}