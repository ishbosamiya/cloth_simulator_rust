#version 330 core

in vec3 in_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
  gl_Position = projection * view * model * vec4(in_pos, 1.0);
}
