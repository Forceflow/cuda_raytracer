#include <GL/glew.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <exception>

// static, we only want this function to be available in this file scope
static std::string loadFileToString(const char *filename){
	std::ifstream file(filename, std::ios::in);
	std::string text;

	if (file){
		file.seekg(0, std::ios::end); // go to end
		text.resize(file.tellg()); // resize text buffer to file size
		file.seekg(0, std::ios::beg); // back to begin
		file.read(&text[0], text.size()); // read into buffer
		file.close();
	}
	else {
		std::string error_message = std::string("File not found: ") + filename;
		fprintf(stderr, error_message.c_str());
		throw std::runtime_error(error_message);
	}

	return text;
}

class GLSLShader{
public:
	GLuint shader;
	GLint compiled;
	GLenum shadertype;
private:
	std::string shader_location;
	std::string shader_source_text;

public:
	GLSLShader::GLSLShader(const std::string shader_location, GLenum shadertype) : shader(0), shader_location(shader_location), compiled(false), shadertype(shadertype) {
		shader_source_text = loadFileToString(shader_location.c_str());
	}

	void GLSLShader::compile(){
		shader = glCreateShader(shadertype);
		const char *c_str = shader_source_text.c_str();
		glShaderSource(shader, 1, &c_str, NULL);
		glCompileShader(shader);

		// check if shader compiled
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

		if (!compiled){
			char temp[256] = "";
			glGetShaderInfoLog(shader, 256, NULL, temp);
			printf("Shader compilation error:\n%s\n%s\n", shader_location, temp);
			glDeleteShader(shader);
			compiled = true;
		}
	}
};

class GLSLProgram{
public:
	GLuint program;
	bool compiled;

private:
	GLSLShader vertex_shader;
	GLSLShader fragment_shader;

public:
	GLSLProgram::GLSLProgram(GLSLShader vertex, GLSLShader fragment) : program(0), vertex_shader(vertex), fragment_shader(fragment), compiled(false) {
	}

	void GLSLProgram::compile(){
		program = glCreateProgram();

		if (vertex_shader.compiled && fragment_shader.compiled){
			glAttachShader(program, vertex_shader.shader);
			glAttachShader(program, fragment_shader.shader);
		}

		glLinkProgram(program);

		int infologLength = 0;
		int charsWritten = 0;

		glGetProgramiv(program, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

		if (infologLength > 0){
			char *infoLog = (char *)malloc(infologLength);
			glGetProgramInfoLog(program, infologLength, (GLsizei *)&charsWritten, infoLog);
			printf("Program compilation error: %s\n", infoLog);
			free(infoLog);
		} else {
			compiled = true;
		}
	}
};