#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

#include "cuda_error_check.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "shader_tools.h"

using namespace std;

// GLFW
GLFWwindow* window;
int width = 512;
int height = 512;

// CUDA OPENGL interop
unsigned int *cuda_dest_resource;
GLuint shDrawTex;  // draws a texture
struct cudaGraphicsResource *cuda_tex_result_resource;
GLuint fbo_source;
struct cudaGraphicsResource *cuda_tex_screen_resource;
unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;
// (offscreen) render target fbo variables
GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result
#ifndef USE_TEXTURE_RGBA8UI
#   pragma message("Note: Using Texture fmt GL_RGBA16F_ARB")
#else
// NOTE: the current issue with regular RGBA8 internal format of textures
// is that HW stores them as BGRA8. Therefore CUDA will see BGRA where users
// expected RGBA8. To prevent this issue, the driver team decided to prevent this to happen
// instead, use RGBA8UI which required the additional work of scaling the fragment shader
// output from 0-1 to 0-255. This is why we have some GLSL code, in this case
#   pragma message("Note: Using Texture RGBA8UI + GLSL for rendering")
#endif
GLuint shDraw;

extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw);

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
	//if (!glewIsSupported(
	//	//"GL_VERSION_2_0"
	//	//"GL_ARB_pixel_buffer_object"
	//	//"GL_EXT_framebuffer_object"
	//	"ARB_depth_clamp"))
	//{
	//	printf("ERROR: Support for necessary OpenGL extensions missing.");
	//	fflush(stderr);
	//	return false;
	//}

	printf("%s\n", glGetString(GL_VERSION));
	printf("%s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

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
	//glsl_drawtex_vertshader_src = loadFileToString("vertex_shader.glsl");

	return true;
}

bool initGLFW(){
	if (!glfwInit()) exit(EXIT_FAILURE);
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	window = glfwCreateWindow(width, height, "Voxel Ray Caster", NULL, NULL);
	if (!window){ glfwTerminate(); exit(EXIT_FAILURE); }
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glfwSetKeyCallback(window, keyboardfunc);

	printf("%i \n", glfwGetWindowAttrib(window, GLFW_OPENGL_PROFILE));
	printf("%i \n", GLFW_OPENGL_COMPAT_PROFILE);
	printf("%i \n", GLFW_OPENGL_CORE_PROFILE);

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