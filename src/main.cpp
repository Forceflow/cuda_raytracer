// OpenGL
#include <GL/glew.h> // Take care: GLEW should be included before GLFW
#include <GLFW/glfw3.h>
// CUDA
#include "cuda_error_check.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// C++ libs
#include <string>
#include "shader_tools.h"
#include "gl_tools.h"
#include "glfw_tools.h"

using namespace std;

// GLFW
GLFWwindow* window;
int width = 512;
int height = 512;

// OpenGL
GLuint shaderProgram;
GLuint VBO, VAO, EBO;

// Cuda <-> OpenGl interop resources
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
GLuint shDraw;
const GLenum fbo_targets[] =
{
	GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
	GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
};

// Shaders from CUDA2GL sample
static const char *glsl_drawtex_vertshader_src =
"void main(void)\n"
"{\n"
"	gl_Position = gl_Vertex;\n"
"	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
"}\n";

static const char *glsl_drawtex_fragshader_src =
"#version 130\n"
"uniform usampler2D texImage;\n"
"void main()\n"
"{\n"
"   vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
"	gl_FragColor = c / 255.0;\n"
"}\n";

static const char *glsl_draw_fragshader_src =
"#version 130\n"
"out uvec4 FragColor;\n"
"void main()\n"
"{"
"  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
"}\n";

// QUAD GEOMETRY
GLfloat vertices[] = {
	// Positions          // Colors           // Texture Coords
	0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // Top Right
	0.5f, -0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // Bottom Right
	-0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // Bottom Left
	-0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // Top Left 
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {  // Note that we start from 0!
	0, 1, 3,  // First Triangle
	1, 2, 3   // Second Triangle
};

void createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
	// create a texture
	glGenTextures(1, tex_cudaResult);
	glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
	SDK_CHECK_ERROR_GL();
	// register this texture with CUDA
	/*checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult,
		GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));*/
}

void display(void){
	glfwPollEvents();
	// Clear the colorbuffer
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // wireframe mode

	glUseProgram(shaderProgram);
	glBindVertexArray(VAO); // binding VAO automatically binds EBO
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0); // unbind VAO

	// Swap the screen buffers
	glfwSwapBuffers(window);
}

// Keyboard
void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods){
	switch (key) {
	}
}

bool initGL(){
	glewExperimental = GL_TRUE;
	glewInit();
	glViewport(0, 0, width, height); // viewport for x,y to normalized device coordinates transformation
	return true;
}

bool initGLFW(){
	if (!glfwInit()) exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(width, height, "The Simplest OpenGL Quad", NULL, NULL);
	if (!window){ glfwTerminate(); exit(EXIT_FAILURE); }
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glfwSetKeyCallback(window, keyboardfunc);
	return true;
}

int main(int argc, char *argv[]) {
	initGLFW();
	initGL();

	printGLFWInfo(window);
	printGlewInfo();
	printGLInfo();

	// Generate buffers
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Compile shaders
	GLSLShader vertex("D:/jeroenb/Implementation/cuda_raytracer/src/vertex_shader.glsl", GL_VERTEX_SHADER);
	vertex.compile();
	GLSLShader fragment("D:/jeroenb/Implementation/cuda_raytracer/src/fragment_shader.glsl", GL_FRAGMENT_SHADER);
	fragment.compile();
	GLSLProgram program(vertex, fragment);
	program.compile();

	shaderProgram = program.program;

	// Buffer setup
	// Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
	glBindVertexArray(VAO); // all next calls wil use this VAO (descriptor for VBO)

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Color attribute (3 floats)
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);
	// Texture attribute (2 floats)
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound 
	// vertex buffer object so afterwards we can safely unbind
	glBindVertexArray(0);

	// Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
	// A VAO stores the glBindBuffer calls when the target is GL_ELEMENT_ARRAY_BUFFER. 
	// This also means it stores its unbind calls so make sure you don't unbind the element array buffer before unbinding your VAO, otherwise it doesn't have an EBO configured.

	while (!glfwWindowShouldClose(window))
	{
		display();
		glfwWaitEvents();
	}
	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}