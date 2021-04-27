#version 330 core

layout (location = 0) in vec3 in_pos;
layout (location = 1) in mat4 in_instance_model;

uniform mat4 view;
uniform mat4 projection;

void main()
{
  gl_Position = projection * view * in_instance_model * vec4(in_pos, 1.0);
}
