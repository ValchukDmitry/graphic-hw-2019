#!/usr/bin/env python3.5
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pywavefront

import numpy as np
import time


def normalize(v):
    norm = np.linalg.norm(v)
    if norm:
        v = list(np.array(v) / norm)
    return v


class View:
    def __init__(self, width, height, g_buffer, light_prog, lights_coor, lights_color):
        self.center = [0, 0, 0]
        self.mouse_pos = [0., 0.]
        self.width = width
        self.height = height
        self.scale = 1.
        self.zoom_speed = 1.05
        self.rotation_speed = 0.2
        self.g_buffer = g_buffer
        self.light_prog = light_prog
        self.cam_position = (0, 0, 100)
        self.x = 0
        self.y = 0
        self.lights_coor = lights_coor
        self.lights_color = lights_color
        self._prepare_fbo()

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
        glRotate(self.rotation_speed * dx, *(0, 1, 0))
        y_rotation = np.dot(glGetDoublev(GL_MODELVIEW_MATRIX), (1, 0, 0, 0))[:-1]
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

    def _prepare_fbo(self):
        gBuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, gBuffer)
        # position color buffer
        self.gPosition = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gPosition)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, self.width, self.height, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.gPosition, 0)
        # normal color buffer
        self.gNormal = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gNormal)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self.gNormal, 0)

        # color + specular color buffer
        self.gAlbedoSpec = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gAlbedoSpec)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, self.gAlbedoSpec, 0)

        # tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
        attachments = [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2]
        glDrawBuffers(3, attachments)

        gPosition_dist = glGetUniformLocation(self.light_prog, "gPosition")
        gNormal_dist = glGetUniformLocation(self.light_prog, "gNormal")
        gAlbedoSpec_dist = glGetUniformLocation(self.light_prog, "gAlbedoSpec")
        glUseProgram(self.light_prog)
        glUniform1i(gPosition_dist, self.gPosition)
        glUniform1i(gNormal_dist, self.gNormal)
        glUniform1i(gAlbedoSpec_dist, self.gAlbedoSpec)
        self.gBuffer = gBuffer

        rboDepth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth)
        # glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, gBuffer)
        glUseProgram(0)

    def draw_quad(self):
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1, 1)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glEnd()
        # if not hasattr(self, 'quadVAO') or self.quadVAO == 0:
        #     quad_verts = np.array([-1.0, 1.0, 0.0, 0.0, 1.0,
        #                            -1.0, -1.0, 0.0, 0.0, 0.0,
        #                            1.0, 1.0, 0.0, 1.0, 1.0,
        #                            1.0, -1.0, 0.0, 1.0, 0.0], dtype='float32')
        #     self.quadVAO = glGenVertexArrays(1)
        #     self.quadVBO = glGenBuffers(1)
        #     glBindVertexArray(self.quadVAO)
        #     glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO)
        #     glBufferData(GL_ARRAY_BUFFER, quad_verts, GL_STATIC_DRAW)
        #     glEnableVertexAttribArray(0)
        #
        #     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, 0)
        #     glEnableVertexAttribArray(1)
        #     glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, 3 * 4)
        # glBindVertexArray(self.quadVAO)
        # glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        # glBindVertexArray(0)

    def draw(self):
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindFramebuffer(GL_FRAMEBUFFER, self.gBuffer)

        glDisable(GL_BLEND)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.g_buffer)
        model_dist = glGetUniformLocation(self.g_buffer, "model")
        proj_dist = glGetUniformLocation(self.g_buffer, "projection")
        view_dist = glGetUniformLocation(self.g_buffer, "view")
        glUniformMatrix4fv(model_dist, 1, GL_FALSE, np.diag([1, 1, 1, 1]))
        glMatrixMode(GL_PROJECTION)
        projection_mat = glGetDoublev(GL_PROJECTION_MATRIX)
        glUniformMatrix4fv(proj_dist, 1, GL_FALSE, projection_mat)
        glMatrixMode(GL_MODELVIEW)
        glUniformMatrix4fv(view_dist, 1, GL_FALSE, glGetDoublev(GL_MODELVIEW_MATRIX))
        display()
        glUseProgram(0)

        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # start lighting
        glUseProgram(self.light_prog)

        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.gBuffer)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST)
        location1 = glGetUniformLocation(self.light_prog, "gPosition")
        location2 = glGetUniformLocation(self.light_prog, "gNormal")
        location3 = glGetUniformLocation(self.light_prog, "gAlbedoSpec")
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.gPosition)
        glUniform1i(location1, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.gNormal)
        glUniform1i(location2, 1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.gAlbedoSpec)
        glUniform1i(location3, 2)

        # glEnable(GL_BLEND)
        #
        # glBlendEquation(GL_FUNC_ADD)
        # glBlendFunc(GL_ONE, GL_ONE)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        for i, light in enumerate(zip(self.lights_coor, self.lights_color)):
            cur_position_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Position")
            cur_color_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Color")
            glUniform3f(cur_position_dist, *light[0])
            glUniform3f(cur_color_dist, *light[1])
            constant = 1.0
            linear = 0.7
            quadratic = 1.8
            cur_linear_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Linear")
            cur_quad_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Quadratic")
            glUniform1f(cur_linear_dist, linear)
            glUniform1f(cur_quad_dist, quadratic)
            maxBrightness = np.max([light[1][0], light[1][1], light[1][2]])
            radius = 100  # (-linear + np.sqrt(linear * linear - 4 * quadratic * (constant - (256.0 / 5.0) * maxBrightness))) / (2.0 * quadratic)
            cur_radius_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Radius")
            glUniform1f(cur_radius_dist, radius)
        cam_dist = glGetUniformLocation(self.light_prog, "viewPos")
        glUniform3f(cam_dist, *self.cam_position)
        self.draw_quad()
        glUseProgram(0)

        # copy to main buffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.gBuffer)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height,
                          GL_DEPTH_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        # glFlush()
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
        gluLookAt(*self.cam_position, *self.center, 0, 1, 0)
        glutReshapeWindow(self.width, self.height)


def load_shader(path, shader_type):
    print(path)
    shader = glCreateShader(shader_type)
    with open(path) as f:
        glShaderSource(shader, f.read())
    glCompileShader(shader)
    return shader


def gen_texture(iters, inner_size=1, dtype=float):
    texture = np.array([1 for i in range(iters * inner_size)], dtype=dtype)
    return texture


VERTEX_FORMATS = {
    'V3F': GL_V3F,
    'C3F_V3F': GL_C3F_V3F,
    'N3F_V3F': GL_N3F_V3F,
    'T2F_V3F': GL_T2F_V3F,
    'T2F_C3F_V3F': GL_T2F_C3F_V3F,
    'T2F_N3F_V3F': GL_T2F_N3F_V3F
}


def display():
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    for material in materials:
        if material.gl_floats is None:
            material.gl_floats = (GLfloat * len(material.vertices))(*material.vertices)
            material.triangle_count = int(len(material.vertices) / material.vertex_size)
            glInterleavedArrays(VERTEX_FORMATS.get(material.vertex_format), 0, material.gl_floats)
        glDrawArrays(GL_TRIANGLES, 0, material.triangle_count)


width = 1000
height = 700
TEXTURE_SIZE = 256

glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(width, height)
glutInit(sys.argv)
glutCreateWindow("Valchuk Dmitry. Deferred Shading")

glClearColor(0, 0, 0, 0)
obj = pywavefront.Wavefront('FinalBaseMesh.obj', create_materials=True)
materials = obj.materials.values()

glEnable(GL_DEPTH_TEST)
glEnable(GL_TEXTURE_2D)
# glDepthFunc(GL_LESS)
glEnableClientState(GL_NORMAL_ARRAY)
glEnableClientState(GL_TEXTURE_COORD_ARRAY)
glEnableClientState(GL_VERTEX_ARRAY)
glEnableClientState(GL_TEXTURE_COORD_ARRAY)

model_view_program = glCreateProgram()
g_buffer = glCreateProgram()
deffered_shading = glCreateProgram()

glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

glAttachShader(g_buffer, load_shader("shaders/g_buffer.vert", GL_VERTEX_SHADER))
glAttachShader(g_buffer, load_shader("shaders/g_buffer.frag", GL_FRAGMENT_SHADER))
glAttachShader(deffered_shading, load_shader("shaders/deffered_shading.vert", GL_VERTEX_SHADER))
glAttachShader(deffered_shading, load_shader("shaders/deffered_shading.frag", GL_FRAGMENT_SHADER))

glLinkProgram(g_buffer)
glLinkProgram(deffered_shading)

# create and attach depth buffer (renderbuffer)
if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE):
    print("Framebuffer not complete!")

lights_count = 150

light_positions = []
light_colors = []
for i in range(lights_count):
    xPos = np.random.rand()
    yPos = np.random.rand()
    zPos = np.random.rand()
    light_positions.append([xPos, yPos, zPos])
    r = np.random.rand()
    g = np.random.rand()
    b = np.random.rand()
    light_colors.append([r, g, b])

view = View(width, height, g_buffer, deffered_shading, light_positions, light_colors)

view.reshape_handler(width, height)
glutDisplayFunc(view.draw)
glutMouseFunc(view.mouse_handler)
glutMotionFunc(view.motion_handler)

glutMainLoop()
