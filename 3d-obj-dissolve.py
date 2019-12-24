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
    def __init__(self, width, height, program_draw, program_shader_geometry, light_prog, params, model, lights):
        self.center = [0, 0, 0]
        self.mouse_pos = [0., 0.]
        self.width = width
        self.height = height
        self.scale = 1.
        self.zoom_speed = 1.05
        self.rotation_speed = 0.2
        self.program_draw = program_draw
        self.program_shader_geometry = program_shader_geometry
        self.light_prog = light_prog
        self.model = model
        self.x = 0
        self.y = 0
        self.gPosition = params[0]
        self.gNormal = params[1]
        self.gAlbedoSpec = params[2]
        self.lights = lights

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
        glUseProgram(self.program_draw)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, len(self.model.texture))
        glMatrixMode(GL_MODELVIEW)
        model_view = glGetDoublev(GL_MODELVIEW_MATRIX)
        glMatrixMode(GL_PROJECTION)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        glUseProgram(0)

        glUseProgram(self.program_shader_geometry)

        # self.model_dist = glGetUniformLocation(self.program_shader_geometry, "model")
        # glUniformMatrix4fv(self.model_dist, 1, GL_FALSE, model_view)
        object_positions = [[-3, -3, -3], [0, -3, -3]]
        for obj in object_positions:
            pass
            # model = glMatrix(1.0)
            # model = glTranslate(model, obj)
            # model_view = glScale(model_view, [0.25, 0.25, 0.25])
            # glUniformMatrix4fv(self.model_dist, 1, GL_FALSE, model_view)
            # model.draw(shaderGeometryPass)
        glUseProgram(0)

        glUseProgram(self.light_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.gPosition)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.gNormal)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.gAlbedoSpec)
        for light in self.lights:
            cur_position_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Position")
            cur_color_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Color")
            glUniform3f(cur_position_dist, *light[0])
            glUniform3f(cur_color_dist, *light[1])
            # update attenuation parameters and calculate radius
            constant = 1.0 # note that we don't send this to the shader, we assume it is always 1.0 (in our case)
            linear = 0.7
            quadratic = 1.8
            cur_linear_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Linear")
            cur_quad_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Quadratic")
            glUniform1f(cur_linear_dist, linear)
            glUniform1f(cur_quad_dist, quadratic)
            # then calculate radius of light volume/sphere
            maxBrightness = np.max([light[1][0], light[1][1], light[1][2]])
            radius = 100#(-linear + np.sqrt(linear * linear - 4 * quadratic * (constant - (256.0 / 5.0) * maxBrightness))) / (2.0 * quadratic)
            cur_radius_dist = glGetUniformLocation(self.light_prog, "lights[" + str(i) + "].Radius")
            glUniform1f(cur_radius_dist, radius)

        glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0) # write to default framebuffer
        # blit to default framebuffer. Note that this may or may not work as the internal formats of both the FBO and default framebuffer have to match.
        # the internal formats are implementation defined. This works on all of my systems, but if it doesn't on yours you'll likely have to write to the
        # depth buffer in another shader stage (or somehow see to match the default framebuffer's internal format with the FBO's internal format).
        glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

        # shaderLightBox.use()
        # shaderLightBox.setMat4("projection", projection);
        # shaderLightBox.setMat4("view", view);
        # for (unsigned int i = 0; i < lightPositions.size(); i++)
        # {
        #     model = glm::mat4(1.0f);
        #     model = glm::translate(model, lightPositions[i]);
        #     model = glm::scale(model, glm::vec3(0.125f));
        #     shaderLightBox.setMat4("model", model);
        #     shaderLightBox.setVec3("lightColor", lightColors[i]);
        #     renderCube();
        # }
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


def load_shader(path, shader_type):
    print(path)
    shader = glCreateShader(shader_type)
    with open(path) as f:
        glShaderSource(shader, f.read())
    glCompileShader(shader)
    return shader


def gen_texture(iters, dtype = float):
    texture = np.array([0 for i in range(iters)], dtype=dtype)
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
glutCreateWindow("Valchuk Dmitry. Deferred Shading")

glClearColor(0, 0, 0, 0)
model = Mesh()
model.load('skull.obj')

glEnable(GL_DEPTH_TEST)
glEnableClientState(GL_NORMAL_ARRAY)
glEnableClientState(GL_TEXTURE_COORD_ARRAY)
glEnableClientState(GL_VERTEX_ARRAY)

# glVertexPointer(3, GL_FLOAT, 0, model.vertices)
# glNormalPointer(GL_FLOAT, 0, model.normals)

glVertexPointer(3, GL_FLOAT, 0, model.vertices)
glNormalPointer(GL_FLOAT, 0, model.normals)

light_prog = glCreateProgram()
model_view_program = glCreateProgram()
shaderGeometryProg = glCreateProgram()
program3 = glCreateProgram()

glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

glBindAttribLocation(model_view_program, 1, "coords")
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, False, 0, model.texture)
tex_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, tex_id)

glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, TEXTURE_SIZE, TEXTURE_SIZE,
                0, GL_RED, GL_FLOAT, gen_texture(TEXTURE_SIZE * TEXTURE_SIZE))
glGenerateMipmap(GL_TEXTURE_2D)


glAttachShader(model_view_program, load_shader("shaders/3d-obj.vert", GL_VERTEX_SHADER))
glAttachShader(model_view_program, load_shader("shaders/3d-obj.frag", GL_FRAGMENT_SHADER))
glAttachShader(light_prog, load_shader("shaders/deferred_light_box.vert", GL_VERTEX_SHADER))
glAttachShader(light_prog, load_shader("shaders/deferred_light_box.frag", GL_FRAGMENT_SHADER))
glAttachShader(shaderGeometryProg, load_shader("shaders/g_buffer.vert", GL_VERTEX_SHADER))
glAttachShader(shaderGeometryProg, load_shader("shaders/g_buffer.frag", GL_FRAGMENT_SHADER))
glAttachShader(program3, load_shader("shaders/deffered_shading.vert", GL_VERTEX_SHADER))
glAttachShader(program3, load_shader("shaders/deffered_shading.frag", GL_FRAGMENT_SHADER))

glLinkProgram(model_view_program)
glLinkProgram(light_prog)
glLinkProgram(shaderGeometryProg)
glLinkProgram(program3)


glUseProgram(shaderGeometryProg)
glUseProgram(0)

gBuffer = glGenFramebuffers(1)
glBindFramebuffer(GL_FRAMEBUFFER, gBuffer)
gPosition = 0
gNormal = 0
gAlbedoSpec = 0
print('hello1')
# position color buffer
gPosition = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, gPosition)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, gen_texture(width * height))
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0)
# normal color buffer
gNormal = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, gNormal)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, gen_texture(width * height))
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0)
print('hello3')
# color + specular color buffer
gAlbedoSpec = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, gAlbedoSpec)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, gen_texture(width * height, np.byte))
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedoSpec, 0)
print('hello4')
# tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
attachments = [ GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2]
glDrawBuffers(3, attachments)
print('hello')
# create and attach depth buffer (renderbuffer)
print('1')
rboDepth = glGenRenderbuffers(1)
print('2')
glBindRenderbuffer(GL_RENDERBUFFER, rboDepth)
print('3')
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
print('4')
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth)
print('5')
glBindFramebuffer(GL_FRAMEBUFFER, 0)

lights_count = 32

light_positions = []
light_colors = []
for i in range(lights_count):
    xPos = ((np.random.rand() % 100) / 100.0) * 6.0 - 3.0
    yPos = ((np.random.rand() % 100) / 100.0) * 6.0 - 4.0
    zPos = ((np.random.rand() % 100) / 100.0) * 6.0 - 3.0
    light_positions.append([xPos, yPos, zPos])
    r = ((np.random.rand() % 100) / 200.0) + 0.5
    g = ((np.random.rand() % 100) / 200.0) + 0.5
    b = ((np.random.rand() % 100) / 200.0) + 0.5
    light_colors.append([r, g, b])

view = View(width, height, model_view_program, shaderGeometryProg, program3, \
    (gPosition, gNormal, gAlbedoSpec), model, zip(light_positions, light_colors))
view.reshape_handler(width, height)
glutDisplayFunc(view.draw)
glutMouseFunc(view.mouse_handler)
glutMotionFunc(view.motion_handler)


glutMainLoop()
# while True:
#     glClearColor(0, 0, 0, 1)
#     matrix = glGetDoublev(GL_MODELVIEW_MATRIX)

#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glBindFramebuffer(GL_FRAMEBUFFER, gBuffer)
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glUseProgram(shaderGeometryProg)
    # model_dist = glGetUniformLocation(shaderGeometryProg, "model")
    # glUniformMatrix4fv(model_dist, model)
    # model = np.random.rand(4, 4)
    # shaderGeometryProg.setMat4("model", model)
    # glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), width / height, 0.1, 100.0)
    # glm::mat4 view = camera.GetViewMatrix()
    # glm::mat4 model = glm::mat4(1.0)


# for (unsigned int i = 0; i < NR_LIGHTS; i++)
# {
#     // calculate slightly random offsets
#     float xPos = ((rand() % 100) / 100.0) * 6.0 - 3.0;
#     float yPos = ((rand() % 100) / 100.0) * 6.0 - 4.0;
#     float zPos = ((rand() % 100) / 100.0) * 6.0 - 3.0;
#     lightPositions.push_back(glm::vec3(xPos, yPos, zPos));
#     // also calculate random color
#     float rColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0
#     float gColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0
#     float bColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0
#     lightColors.push_back(glm::vec3(rColor, gColor, bColor));
# }