#!/usr/bin/env python3.5
from OpenGL.GL import *
from OpenGL.GLUT import *


class Slider:
    def __init__(self, coord, width, start_position, text, cur_value_func, view, callback):
        self.view = view
        self.coord = view.from_screen_coords(*coord)
        self.width = width
        self.slider_position = start_position
        self.text = text
        self.cur_value = cur_value_func
        self.callback = callback
        self.quad_width = 0.012

    # returns vertices coords from upper left counterclock-wise
    def get_quad_coords(self):
        return [(self.coord[0] + self.slider_position * self.width - self.quad_width,
                 self.coord[1] - self.quad_width),
                (self.coord[0] + self.slider_position * self.width - self.quad_width,
                    self.coord[1] + self.quad_width),
                (self.coord[0] + self.slider_position * self.width + self.quad_width,
                    self.coord[1] + self.quad_width),
                (self.coord[0] + self.slider_position * self.width + self.quad_width,
                    self.coord[1] - self.quad_width)]

    def is_slider_click(self, x, y):
        x, y = view.from_screen_coords(x, y)
        coords = self.get_quad_coords()
        return coords[0][1] <= y <= coords[1][1] and coords[1][0] <= x <= coords[2][0]

    def move_slider(self, x, y):
        x, y = view.from_screen_coords(x, y)
        self.slider_position = (x - self.coord[0]) / self.width
        if self.slider_position > 1:
            self.slider_position = 1
        if self.slider_position < 0:
            self.slider_position = 0
        self.callback(self.slider_position)

    def draw(self):
        glColor3f(0.28, 0, 0)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glVertex2f(self.coord[0], self.coord[1])
        glVertex2f(self.coord[0] + self.width, self.coord[1])
        glEnd()
        glColor3f(1, 0.5, 0.)
        glBegin(GL_QUADS)
        for coords in self.get_quad_coords():
            glVertex2f(coords[0], coords[1])
        glEnd()

        glRasterPos2f(self.coord[0], self.coord[1] + 0.03)
        for c in self.text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(c))

        glRasterPos2f(self.coord[0] + self.width + 0.03,
                      self.coord[1] - 0.015)
        formated_number = ""
        if isinstance(self.cur_value(), int):
            formated_number = '{}'.format(self.cur_value())
        if isinstance(self.cur_value(), float):
            formated_number = '{:.2f}'.format(self.cur_value())
        for c in formated_number:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(c))


class View:
    def __init__(self, width, height, program):
        self.center = [0.5, 0.]
        self.mouse_pos = [0., 0.]
        self.width = width
        self.height = height
        self.scale = 1.
        self.threshold = 8.
        self.iterations = 100
        self.zoom_speed = 0.05
        self.slider = None
        self.program = program
        self.sliders = [
            Slider((60, 60), 0.5, 0.1, "Max iterations",
                   lambda: self.iterations, self, self.change_iterations),
            Slider((60, 120), 0.5, 0.2, "Threshold",
                   lambda: self.threshold, self, self.change_threshold),
        ]

    def from_screen_coords(self, x, y):
        return (2 * x - self.width) / self.width, -(2 * y - self.height) / self.height

    def to_screen_coords(self, x, y):
        return (x * self.width + self.width) / 2, -(y * self.width + self.width) / 2

    def change_iterations(self, slider_position):
        self.iterations = int(slider_position * 1000)

    def change_threshold(self, slider_position):
        self.threshold = slider_position * 40

    def mouse_handler(self, btn, state, x, y):
        GLUT_WHEEL_UP = 3
        GLUT_WHEEL_DOWN = 4
        self.mouse_pos[0], self.mouse_pos[1] = self.from_screen_coords(x, y)

        if btn == GLUT_WHEEL_UP or btn == GLUT_WHEEL_DOWN:
            self.zoom(1 if btn == GLUT_WHEEL_UP else -1, x, y)

        self.slider = None
        if btn == GLUT_LEFT_BUTTON:
            for slider in self.sliders:
                if slider.is_slider_click(x, y):
                    self.slider = slider

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glUseProgram(self.program)
        glProgramUniform1i(self.program, glGetUniformLocation(
            self.program, "iterations"), self.iterations)
        glProgramUniform1f(self.program, glGetUniformLocation(
            self.program, "threshold"), self.threshold)
        glProgramUniform2f(self.program, glGetUniformLocation(
            self.program, "center"), self.center[0], self.center[1])
        glProgramUniform1f(self.program, glGetUniformLocation(
            self.program, "scale"), self.scale)

        glBegin(GL_QUADS)
        glTexCoord2f(-1, -1)
        glVertex2f(-1, -1)
        glTexCoord2f(1, -1)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glTexCoord2f(-1, 1)
        glVertex2f(-1, 1)
        glEnd()
        glUseProgram(0)
        for slider in self.sliders:
            slider.draw()
        glutSwapBuffers()

    def motion_handler(self, x, y):
        screen_x, screen_y = self.from_screen_coords(x, y)

        if not self.slider:
            self.center[0] += (screen_x -
                               self.mouse_pos[0]) * self.scale
            self.center[1] += (screen_y -
                               self.mouse_pos[1]) * self.scale

        if self.slider:
            self.slider.move_slider(x, y)
        self.mouse_pos[0] = screen_x
        self.mouse_pos[1] = screen_y
        glutPostRedisplay()

    def zoom(self, direction, x, y):
        x, y = self.from_screen_coords(x, y)

        zoom_factor = 1 + (-self.zoom_speed if direction >
                           0 else self.zoom_speed)

        self.center[0] += x * self.scale * (zoom_factor - 1)
        self.center[1] += y * self.scale * (zoom_factor - 1)

        self.scale *= zoom_factor

    def reshape_handler(self, w, h):
        glutReshapeWindow(self.width, self.height)


def load_shader(path, shader_type):
    shader = glCreateShader(shader_type)
    with open(path) as f:
        glShaderSource(shader, f.read())
    glCompileShader(shader)
    return shader

def gen_texture(iters):
    r, g, b = 0., 0., 0.
    texture = []
    for _ in range(iters):
        texture.append(r)
        texture.append(g)
        texture.append(b)
        r += 2. / iters
        g += 1. / iters
        if r > 1.:
            r = 1.
        if g > 0.6:
            g = 0.6

    for _ in range(iters):
        texture.append(r)
        texture.append(g)
        texture.append(b)
        r -= 1. / iters
        g -= 2. / iters
        if r < 0:
            r = 0.
        if g < 0:
            g = 0.

    return texture

width = 1000
height = 1000

glutInit()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
glutInitWindowSize(width, height)

glutCreateWindow(b"Valchuk Dmitry Mandelbrot")

program = glCreateProgram()
view = View(width, height, program)

glutDisplayFunc(view.draw)
glutIdleFunc(glutPostRedisplay)
glutMouseFunc(view.mouse_handler)
glutMotionFunc(view.motion_handler)
glutReshapeFunc(view.reshape_handler)

texture = gen_texture(100)

glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, len(
    texture) / 3, 0, GL_RGB, GL_FLOAT, texture)

glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

glAttachShader(program, load_shader("mandelbrot.frag", GL_FRAGMENT_SHADER))

glLinkProgram(program)

glutMainLoop()
