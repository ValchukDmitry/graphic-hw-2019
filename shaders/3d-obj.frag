#version 130

varying vec3 normal;
varying vec3 camera_position;

uniform vec4 start = vec4(0.4, 0.4, 0.45, 1);

out vec3 gPosition

void main() {
    float light_force = 0.5 * dot(normal, camera_position);
    gl_FragColor = light_force * start;
}