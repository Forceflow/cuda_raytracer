#version 130
uniform usampler2D texImage;
void main()
{
   vec4 c = texture(texImage, gl_TexCoord[0].xy);
   gl_FragColor = c / 255.0;
}