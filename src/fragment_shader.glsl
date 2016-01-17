#version 130
out uvec4 FragColor;
void main()
{
  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);
}