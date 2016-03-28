#version 130

// beter: vertices in VBO rammen en hier - layout(location = 0) in vec4 position;

void main(void)
{
	gl_Position = gl_Vertex;
	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;
}

// gl_Position is a builtin
// gl_TexCoord is deprecated