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
int WIDTH = 256;
int HEIGHT = 256;

// OpenGL
GLuint shaderProgram;
GLuint VBO, VAO, EBO;
GLSLShader draw_f; // GLSL fragment shader
GLSLShader drawtex_f; // GLSL fragment shader
GLSLShader drawtex_v; // GLSL fragment shader
GLSLProgram shdraw; // GLSL program to draw
GLSLProgram shdrawtex; // GLSLS program for textured draw

// Cuda <-> OpenGl interop resources
unsigned int* cuda_dest_resource;
struct cudaGraphicsResource* cuda_tex_result_resource;

extern "C" void
launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw);

GLuint fbo_source;
struct cudaGraphicsResource *cuda_tex_screen_resource;
size_t size_tex_data;
unsigned int num_texels;
unsigned int num_values;
// (offscreen) render target fbo variables
GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // OpenGL Texture for cuda result

const GLenum fbo_targets[] =
{
	GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
	GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
};

// Shaders from CUDA2GL sample
static const char *glsl_drawtex_vertshader_src =
"#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec3 color;\n"
"layout (location = 2) in vec2 texCoord;\n"
"\n"
"out vec3 ourColor;\n"
"out vec2 ourTexCoord;\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0f);\n"
"	ourColor = color;\n"
"	ourTexCoord = texCoord;\n"
"}\n";

static const char *glsl_drawtex_fragshader_src =
"#version 330 core\n"
"uniform usampler2D texImage;\n"
"in vec3 ourColor;\n"
"in vec2 ourTexCoord;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"   vec4 c = texture(texImage, ourTexCoord);\n"
"	color = (c / 255.0);\n"
"}\n";

//static const char *glsl_draw_fragshader_src =
//"#version 130\n"
//"out uvec4 FragColor;\n"
//"void main()\n"
//"{"
//"  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
//"}\n";

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

// Create OpenGL texture and bind it to CUDA
void createTextureDst(GLuint* tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
	// create an OpenGL texture
	glGenTextures(1, tex_cudaResult); // generate 1 texture
	glBindTexture(GL_TEXTURE_2D, *tex_cudaResult); // set it as current target
	// set basic texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// Specify 2D texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
	SDK_CHECK_ERROR_GL();
	// Register this texture with CUDA
	HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void initGLBuffers()
{
	// create texture that will receive the result of cuda
	createTextureDst(&tex_cudaResult, WIDTH, HEIGHT);
	// create shader program
	drawtex_v = GLSLShader("Textured draw vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
	drawtex_f = GLSLShader("Textured draw fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
	shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
	shdrawtex.compile();
	SDK_CHECK_ERROR_GL();
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
	glewExperimental = GL_TRUE; // need this to enforce core profile
	glewInit(); // this causes enum error
	glViewport(0, 0, WIDTH, HEIGHT); // viewport for x,y to normalized device coordinates transformation
	SDK_CHECK_ERROR_GL();
	return true;
}

void initCUDABuffers()
{
	// set up vertex data parameter
	num_texels = WIDTH * WIDTH;
	num_values = num_texels * 4;
	size_tex_data = sizeof(GLubyte) * num_values;
	HANDLE_CUDA_ERROR(cudaMalloc(&cuda_dest_resource, size_tex_data)); // Allocate CUDA memory for color output
}

bool initGLFW(){
	if (!glfwInit()) exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(WIDTH, WIDTH, "The Simplest OpenGL Quad", NULL, NULL);
	if (!window){ glfwTerminate(); exit(EXIT_FAILURE); }
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glfwSetKeyCallback(window, keyboardfunc);
	return true;
}

void generateCUDAImage()
{
	// run the Cuda kernel
	// calculate grid size
	dim3 block(16, 16, 1);
	dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1); // 2D grid, every thread will compute a pixel
	launch_cudaRender(grid, block, 0, cuda_dest_resource, WIDTH); // launch with 0 additional shared memory allocated
	// We want to copy cuda_dest_resource data to the texture
	// map buffer objects to get CUDA device pointers
	cudaArray *texture_ptr;
	HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
	HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));

	int num_texels = WIDTH * HEIGHT;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;
	CHECK_CUDA_ERROR(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));
	CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));
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

	checkCudaRequirements();

	initCUDABuffers();
	initGLBuffers();
	
	generateCUDAImage();

	std::string vertexsrc = loadFileToString("D:/jeroenb/Implementation/cuda_raytracer/src/vertex_shader.glsl");
	GLSLShader vertex(std::string("Vertex shader"), vertexsrc.c_str(), GL_VERTEX_SHADER);
	vertex.compile();
	std::string fragmentsrc = loadFileToString("D:/jeroenb/Implementation/cuda_raytracer/src/fragment_shader.glsl");
	GLSLShader fragment(std::string("Fragment shader"), fragmentsrc.c_str(), GL_FRAGMENT_SHADER);
	fragment.compile();
	GLSLProgram program(&vertex, &fragment);
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
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound 
	// vertex buffer object so afterwards we can safely unbind
	glBindVertexArray(0);

	// Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
	// A VAO stores the glBindBuffer calls when the target is GL_ELEMENT_ARRAY_BUFFER. 
	// This also means it stores its unbind calls so make sure you don't unbind the element array buffer before unbinding your VAO, otherwise it doesn't have an EBO configured.

	//initGLBuffers();

	while (!glfwWindowShouldClose(window))
	{
		display();
		glfwWaitEvents();
	}
	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}