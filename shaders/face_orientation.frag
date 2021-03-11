#version 330 core

out vec4 FragColor;

void main()
{
  FragColor = gl_FrontFacing ? vec4(0.0, 0.3, 1.0, 1.0) : vec4(1.0, 0.3, 0.0, 1.0);
}
