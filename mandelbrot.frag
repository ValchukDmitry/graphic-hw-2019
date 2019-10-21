uniform sampler1D texture;
uniform int iterations;
uniform float threshold;
uniform vec2 center;
uniform float scale;

void main() {
    vec2 prev, c;
    c.x = gl_TexCoord[0].x * scale - center.x;
    c.y = gl_TexCoord[0].y * scale - center.y;
    prev = c;
    int iter = 0;
    while(iter < iterations) {
        float x = prev.x * prev.x - prev.y * prev.y + c.x;
        float y = 2 * prev.y * prev.x + c.y;
        prev.x = x;
        prev.y = y;
        if (x * x + y * y > threshold) {
            break;
        }
        iter++;
    }
    if(iter == iterations) {
        gl_FragColor = texture1D(texture, 0.);
    } else {
        gl_FragColor = texture1D(texture, (1. * iter) / iterations);
    }
}