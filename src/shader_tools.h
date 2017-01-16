#include <GL/glew.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <exception>

// static, we only want this function to be available in this file scope
inline std::string loadFileToString(const char *filename){
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
	const char* shader_src;

public:
	GLSLShader::GLSLShader(std::string shader_name, const char *shader_text, GLenum shadertype) : 
		shader(0), compiled(false), shadertype(shadertype), shader_name(shader_name), shader_src(shader_text){
	}

	void GLSLShader::compile(){
		shader = glCreateShader(shadertype);
		glShaderSource(shader, 1, &shader_src, NULL);
		glCompileShader(shader);

		// check if shader compiled
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

		if (!compiled){
			char temp[256] = "";
			glGetShaderInfoLog(shader, 256, NULL, temp);
			printf("Shader compilation error:\n%s\n", temp);
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
	GLSLShader* vertex_shader;
	GLSLShader* fragment_shader;

public:
	GLSLProgram::GLSLProgram(GLSLShader* vertex, GLSLShader* fragment) : program(0), vertex_shader(vertex), fragment_shader(fragment), compiled(false) {
	}

	void GLSLProgram::compile(){
		program = glCreateProgram();

		// try to attach all shaders
		GLSLShader* shaders[2] = {vertex_shader, fragment_shader};
		for (unsigned int i = 0; i < 2; i++) {
			if (shaders[i] != NULL) {
				if (!shaders[i]->compiled) {shaders[i]->compile();} // compile shader if not yet compiled
				glAttachShader(program, shaders[i]->shader);
				printf("Attached shader: %s. \n", shaders[i]->shader_name.c_str());
			}
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