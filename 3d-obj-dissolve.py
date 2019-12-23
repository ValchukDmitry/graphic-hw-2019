#!/usr/bin/env python3.5
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import time


class Mesh:
    def __init__(self):
        self.vertices = []
        self.texture = []
        self.normals = []
        self.vertex_indices = []
        self.texture_indices = []
        self.normal_indices = []


    def get_face_elem(self, token, vertex_indices, texture_indices, normal_indices):
        parts = token.split("/")
        parts_n = len(parts)
        vertex_indices.append(int(parts[0]) - 1)
        if parts_n >= 2 and parts[1]:
            texture_indices.append(int(parts[1]) - 1)
        if parts_n >= 3 and parts[2]:
            normal_indices.append(int(parts[2]) - 1)


    def get_face(self, data):
        vertex_indices = []
        texture_indices = []
        normal_indices = []
        for i in range(1, 4):
            self.get_face_elem(data[i], vertex_indices, texture_indices, normal_indices)
        if len(data) >= 5:
            for i in [1, 3, 4]:
                self.get_face_elem(data[i], vertex_indices, texture_indices, normal_indices)
        return vertex_indices, texture_indices, normal_indices


    def load(self, filename):
        with open(filename) as file:
            for line in file:
                line = line.replace("\\", " ")
                data = line.split()
                if not data:
                    continue
                if data[0] == "v":
                    self.vertices.append([float(x) for x in data[1:(3 + 1)]])
                elif data[0] == "vn":
                    self.normals.append([float(x) for x in data[1:(3 + 1)]])
                elif data[0] == "vt":
                    self.texture.append([float(x) for x in data[1:(2 + 1)]])
                elif data[0] == "f":
                    vertex_indices, texture_indices, normal_indices = self.get_face(data)
                    self.vertex_indices.extend(vertex_indices)
                    self.texture_indices.extend(texture_indices)
                    self.normal_indices.extend(normal_indices)
        self.vertices = [self.vertices[ind] for ind in self.vertex_indices]
        if self.normals:
            self.normals = [self.normals[ind] for ind in self.normal_indices]
        else:
            self.normals = []
            for i in range(0, len(self.vertices), 3):
                mat = [self.vertices[i + j].copy() for j in range(3)]
                ab = np.subtract(mat[1], mat[0])
                ac = np.subtract(mat[2], mat[0])
                self.normals.extend([list(normalize(np.cross(ab, ac)))] * 3)
        self.texture = [self.texture[ind] for ind in self.texture_indices] if self.texture else [1 for _ in self.vertices]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm:
        v = list(np.array(v) / norm)
    return v


class View:
    def __init__(self, width, height, program, model, threshold_start, threshold_end):
        self.center = [0, 0, 0]
        self.mouse_pos = [0., 0.]
        self.width = width
        self.height = height
        self.scale = 1.
        self.zoom_speed = 1.05
        self.rotation_speed = 0.2
        self.program = program
        self.model = model
        self.x = 0
        self.y = 0
        animation_time = 200
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.start_time = time.time()

        self.thresholds = \
                [i for i in np.arange(0, 1, 1/animation_time)] + \
                [i for i in np.arange(0, 1, 1/animation_time)][::-1]


    def from_screen_coords(self, x, y):
        return (2 * x - self.width) / self.width, -(2 * y - self.height) / self.height


    def mouse_handler(self, button, state, x, y):
        GLUT_WHEEL_UP = 3
        GLUT_WHEEL_DOWN = 4
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            self.x = x
            self.y = y
        else:
            self.x = -1
            self.y = -1
        if button == GLUT_WHEEL_UP and state == GLUT_UP:
            glMatrixMode(GL_MODELVIEW)
            glScale(self.zoom_speed, self.zoom_speed, self.zoom_speed)
            glutPostRedisplay()
        elif button == GLUT_WHEEL_DOWN and state == GLUT_DOWN:
            glMatrixMode(GL_MODELVIEW)
            glScale(1 / self.zoom_speed, 1 / self.zoom_speed, 1 / self.zoom_speed)
            glutPostRedisplay()


    def rotate_camera(self, dx, dy):
        glMatrixMode(GL_MODELVIEW)
        glRotate(self.rotation_speed * dx, *(0, 0, 1))
        y_rotation = np.dot(glGetDoublev(GL_MODELVIEW_MATRIX), (0, 1, 0, 0))[:-1]
        glRotate(self.rotation_speed * dy, *y_rotation)
        glutPostRedisplay()


    def motion_handler(self, x, y):
        if self.x >= 0 and self.y >= 0:
            dx = x - self.x
            dy = y - self.y
            self.rotate_camera(dx, dy)
            self.x = x
            self.y = y
            glutPostRedisplay()


    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, len(model.texture))
        glutSwapBuffers()


    def zoom(self, direction, x, y):
        x, y = self.from_screen_coords(x, y)
        zoom_factor = 1 + (-self.zoom_speed if direction >
                           0 else self.zoom_speed)
        glMatrixMode(GL_MODELVIEW)
        glScale(zoom_factor, zoom_factor, zoom_factor)
        glutPostRedisplay()


    def reshape_handler(self, w, h):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glViewport(0, 0, width, height)
        gluPerspective(45, width / height, 0.001, 200)
        gluLookAt(100, 0, 0, *self.center, 0, 0, 1)
        glutReshapeWindow(self.width, self.height)


    def animate_handler(self):
        start = self.thresholds[int((self.start_time - time.time()) * 30) % len(self.thresholds)]
        end = start - 0.1
        glUniform1f(self.threshold_start, start)
        glUniform1f(self.threshold_end, end)
        glutPostRedisplay()

def load_shader(path, shader_type):
    shader = glCreateShader(shader_type)
    with open(path) as f:
        glShaderSource(shader, f.read())
    glCompileShader(shader)
    return shader


def gen_texture(iters):
    texture = np.array([i * (np.random.random() + 0.5) for i in range(TEXTURE_SIZE*TEXTURE_SIZE)], dtype=float)
    texture -= np.min(texture)
    if np.max(texture):
        texture /= np.max(texture)
    return list(texture)


width = 1000
height = 700
TEXTURE_SIZE = 256

glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(width, height)
glutInit(sys.argv)
glutCreateWindow("Valchuk Dmitry. 3D-Object")

glClearColor(0, 0, 0, 0)
model = Mesh()
model.load('skull.obj')
glEnable(GL_DEPTH_TEST)
glEnableClientState(GL_NORMAL_ARRAY)
glEnableClientState(GL_TEXTURE_COORD_ARRAY)
glEnableClientState(GL_VERTEX_ARRAY)

glVertexPointer(3, GL_FLOAT, 0, model.vertices)
glNormalPointer(GL_FLOAT, 0, model.normals)


program = glCreateProgram()
glAttachShader(program, load_shader("3d-obj-dissolve.vert", GL_VERTEX_SHADER))
glAttachShader(program, load_shader("3d-obj-dissolve.frag", GL_FRAGMENT_SHADER))
glLinkProgram(program)
glUseProgram(program)

glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

glBindAttribLocation(program, 1, "coords")
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, False, 0, model.texture)
tex_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, tex_id)

glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, TEXTURE_SIZE, TEXTURE_SIZE,
                0, GL_RED, GL_FLOAT, gen_texture(TEXTURE_SIZE * TEXTURE_SIZE))
glGenerateMipmap(GL_TEXTURE_2D)

threshold_start = glGetUniformLocation(program, "threshold_start") if model.texture else -1
threshold_end = glGetUniformLocation(program, "threshold_end") if model.texture else -1

view = View(width, height, program, model, threshold_start, threshold_end)
view.reshape_handler(width, height)
glutDisplayFunc(view.draw)
glutMouseFunc(view.mouse_handler)
glutMotionFunc(view.motion_handler)
glutIdleFunc(view.animate_handler)
lights_loc = glGetUniformLocation(program, "lights")


glutReshapeFunc(view.reshape_handler)

glLinkProgram(program)

glutMainLoop()