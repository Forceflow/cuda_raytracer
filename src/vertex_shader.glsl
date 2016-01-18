#version 130
void main(void)
{
	gl_Position = gl_Vertex;
	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;
}