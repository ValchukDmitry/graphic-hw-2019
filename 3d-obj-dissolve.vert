#version 130

varying vec3 normal;
varying vec3 camera_position;

in vec2 coords;
out vec2 fragment_tex_positon;

void main() {
    camera_position = normalize(vec3(gl_ModelViewMatrix * gl_Vertex));
    normal = normalize(gl_NormalMatrix * gl_Normal);
    gl_Position = ftransform();

    fragment_tex_positon = coords;
}