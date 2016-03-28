#version 130
uniform usampler2D texImage;
out vec4 fragcolor; // only one output, so assigned to 0 (automatically) (otherwise use
void main()
{
    vec4 c = texture(texImage, gl_TexCoord[0].xy);
	fragcolor = c / 255.0;
}
