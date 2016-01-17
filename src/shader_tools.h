#include <GL/glew.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <exception>

// static, we only want this function to be available in this file scope
static std::string loadFileToString(const char* const filename){
	std::ifstream file(filename, std::ios::in);
	std::string text;

	if (file){
		file.seekg(0, std::ios::end); // go to end
		text.resize(file.tellg()); // resize text buffer to file size
		file.seekg(0, std::ios::beg); // back to begin
		file.read(&text[0], text.size()); // read into buffer
		file.close();
	} else {
		std::string error_message = std::string("File not found: ") + filename;
		fprintf(stderr, error_message.c_str());
		throw std::runtime_error(error_message);
	}

	return text;
}

GLuint compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src)
{
	GLuint v, f, p = 0;

	p = glCreateProgram();

	if (vertex_shader_src){
		v = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(v, 1, &vertex_shader_src, NULL);
		glCompileShader(v);

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

		if (!compiled){
			char temp[256] = "";
			glGetShaderInfoLog(v, 256, NULL, temp);
			printf("Vtx Compile failed:\n%s\n", temp);
			glDeleteShader(v);
			return 0;
		}
		else{
			glAttachShader(p, v);
		}
	}

	if (fragment_shader_src){
		f = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(f, 1, &fragment_shader_src, NULL);
		glCompileShader(f);

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

		if (!compiled){
			char temp[256] = "";
			glGetShaderInfoLog(f, 256, NULL, temp);
			printf("frag Compile failed:\n%s\n", temp);
			glDeleteShader(f);
			return 0;
		}
		else{
			glAttachShader(p, f);
		}
	}

	glLinkProgram(p);

	int infologLength = 0;
	int charsWritten = 0;

	glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

	if (infologLength > 0){
		char *infoLog = (char *)malloc(infologLength);
		glGetProgramInfoLog(p, infologLength, (GLsizei *)&charsWritten, infoLog);
		printf("Shader compilation error: %s\n", infoLog);
		free(infoLog);
	}

	return p;
}

//GLuint LoadShader(GLenum eShaderType, const char* const filename)
//{
//	std::string shader_source = loadFileToString(filename);
//	GLuint shader = glCreateShader(eShaderType);
//
//
//	try
//	{
//		return glutil::CompileShader(eShaderType, shaderData.str());
//	}
//	catch (std::exception &e)
//	{
//		fprintf(stderr, "%s\n", e.what());
//		throw;
//	}
//}
