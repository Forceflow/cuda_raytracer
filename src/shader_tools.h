#include <GL/glew.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <exception>

// Simple helper to switch between character arrays and C++ strings 
struct ShaderStringHelper{
	const char *p;
	ShaderStringHelper(const std::string& s) : p(s.c_str()) {}
	operator const char**() { return &p; }
};

// Function to load text from file
// static, we only want this function to be available in this file's scope
inline static std::string loadFileToString(const char *filename){
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
	std::string shader_name;
private:
	std::string shader_src; // internal string representation of shader

public:
	GLSLShader::GLSLShader() :
		shader(0), compiled(false), shadertype(0), shader_name(""), shader_src("") {
	}

	GLSLShader::GLSLShader(std::string shader_name, const char *shader_text, GLenum shadertype) : 
		shader(0), compiled(false), shadertype(shadertype), shader_name(shader_name), shader_src(std::string(shader_text)){
	}

	GLSLShader::GLSLShader(std::string shader_name, std::string shader_text, GLenum shadertype) :
		shader(0), compiled(false), shadertype(shadertype), shader_name(shader_name), shader_src(shader_text) {
	}
	std::string GLSLShader::getSrc() const {
		return shader_src;
	}

	void GLSLShader::setSrc(std::string new_source) {
		shader_src = new_source;
		compiled = false; // setting new source forces recompile
	}

	void GLSLShader::setSrc(const char* new_source) {
		shader_src = std::string(new_source);
		compiled = false; // setting new source forces recompile
	}

	void GLSLShader::compile(){
		shader = glCreateShader(shadertype);
		glShaderSource(shader, 1, ShaderStringHelper(shader_src), NULL);
		glCompileShader(shader);
		// check if shader compiled
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
		if (!compiled){
			getCompilationError(shader);
			glDeleteShader(shader);
			compiled = false;
		}
		else {
			printf("(S) Compiled shader: \"%s\" (%i) \n", shader_name.c_str(), shader);
		}
	}

private:
	void GLSLShader::getCompilationError(GLuint shader) {
		int infologLength = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);
		char* infoLog = (char *)malloc(infologLength);
		glGetShaderInfoLog(shader, infologLength, NULL, infoLog); // will include terminate char
		printf("(S) Shader compilation error:\n%s\n", infoLog);
		free(infoLog);
	}
};

class GLSLProgram{
public:
	GLuint program;
	bool linked;

private:
	GLSLShader* vertex_shader;
	GLSLShader* fragment_shader;

public:
	GLSLProgram::GLSLProgram() : program(0), vertex_shader(NULL), fragment_shader(NULL), linked(false) {
	}

	GLSLProgram::GLSLProgram(GLSLShader* vertex, GLSLShader* fragment) : program(0), vertex_shader(vertex), fragment_shader(fragment), linked(false) {
	}

	void GLSLProgram::compile(){
		// create empty program
		program = glCreateProgram(); 
		// try to attach all shaders
		GLSLShader* shaders[2] = {vertex_shader, fragment_shader};
		for (unsigned int i = 0; i < 2; i++) {
			if (shaders[i] != NULL) {
				if (!shaders[i]->compiled) {shaders[i]->compile();} // try to compile shader if not yet compiled
				if (shaders[i]->compiled) {
					glAttachShader(program, shaders[i]->shader);
					printf("(P) Attached shader \"%s\" (%i) to program \n", shaders[i]->shader_name.c_str(), shaders[i]->shader);
				}
				else {
					printf("(P) Failed to attach shader \"%s\" (%i) to program \n", shaders[i]->shader_name.c_str(), shaders[i]->shader);
					glDeleteProgram(program);
					return;
				}
			}
		}
		// try to link program
		glLinkProgram(program);
		GLint isLinked = 0;
		glGetProgramiv(program, GL_LINK_STATUS, &isLinked); // check if program linked
		if (isLinked == GL_FALSE){
			printLinkError(program);
			glDeleteProgram(program);
			linked = false;
		} else {
			linked = true;
			printf("(P) Linked program %i \n", program);
		}
	}

private:
	void GLSLProgram::printLinkError(GLuint program) {
		GLint infologLength = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);
		char* infoLog = (char *)malloc(infologLength);
		glGetProgramInfoLog(program, infologLength, NULL, infoLog); // will include terminate char
		printf("(P) Program compilation error: %s\n", infoLog);
		free(infoLog);
	}
};