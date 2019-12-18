#version 130

varying vec3 normal;
varying vec3 camera_position;

uniform sampler2D thresh_map;
uniform float threshold_start = 1;
uniform float threshold_end = 1;

uniform vec4 start = vec4(0.4, 0.4, 0.45, 1);
uniform vec4 end = vec4(0.6, 0.25, 0.15, 1);

in vec2 fragment_tex_positon;

void main() {
    if (texture(thresh_map, fragment_tex_positon)[0] < threshold_end) {
        discard;
    }
    if (texture(thresh_map, fragment_tex_positon)[0] < threshold_start) {
        float light_force = 0.5 * dot(normal, camera_position);
        float dist = threshold_start - threshold_end;
        float force = (texture(thresh_map, fragment_tex_positon)[0] - threshold_end) / dist;
        gl_FragColor = light_force * (force * start + (1 - force) * end);
    } else {
        float light_force = 0.5 * dot(normal, camera_position);
        gl_FragColor = light_force * start;
    }
}
