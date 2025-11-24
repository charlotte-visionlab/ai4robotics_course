"""
2D rendering framework adapted from OpenAI gym: https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
Converted to modern pyglet.
"""
# import math
import numpy as np
# import pyglet
from pyglet import gl
# from pyglet.gl import glLineWidth
from pyglet.math import Mat4, Vec3
from pyglet import math as pm
from pyglet import shapes

import pyglet
from pyglet.math import Mat4
from math import radians, cos, sin

RAD2DEG = 180.0 / 3.141592653589793
PI_LOCAL = 3.14159265359


class Viewer(object):
    def __init__(self, width, height, display=None):
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=None, resizable=False)
        self.window.on_close = self.window_closed_by_user
        self.batch = pyglet.graphics.Batch()
        self.geoms = []
        self.onetime_geoms = []
        self.sprites = []

        # Modern pyglet uses a Projection class for transforms
        # We start with an identity transform
        self.transform = Transform()

        # Enable blending for transparency
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        # Use pyglet's built-in camera/viewport handling for setting bounds
        # self.window.projection = pm.Mat4.orthogonal_projection(left, right, bottom, top, -1, 1)
        scale_x = self.width / (right - left)
        scale_y = self.height / (top - bottom)
        # self.transform = Transform(translation=(-left * scale_x, -bottom * scale_y), scale=(scale_x, scale_y))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def add_sprite(self, sprite):
        self.sprites.append(sprite)

    def render(self, return_rgb_array=False):
        gl.glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        # self.transform.enable()
        # Render persistent geoms
        for geom in self.geoms:
            geom.render()

        # Render one-time geoms
        for geom in self.onetime_geoms:
            geom.render()
        # self.transform.disable()

        # Sprites are rendered after all batched items in this structure
        for sprite in self.sprites:
            sprite.draw()

        arr = None
        if return_rgb_array:
            # Modern method for capturing frame buffer
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(image_data.height, image_data.width, 4)
            # Clip to 3 channels and flip vertically
            arr = arr[::-1, :, 0:3]

        self.window.flip()
        self.onetime_geoms = []
        return arr

    # Convenience methods adapted to use pyglet.shapes
    # def draw_circle(self, radius=10, res=30, filled=True, **attrs):
    #     geom = make_circle(radius=radius, res=res, filled=filled)
    #     _add_attrs(geom, attrs)
    #     self.add_onetime(geom)
    #     return geom
    #
    # def draw_polygon(self, v, filled=True, **attrs):
    #     geom = make_polygon(v=v, filled=filled)
    #     _add_attrs(geom, attrs)
    #     self.add_onetime(geom)
    #     return geom
    #
    # def draw_polyline(self, v, **attrs):
    #     geom = make_polyline(v=v)
    #     _add_attrs(geom, attrs)
    #     self.add_onetime(geom)
    #     return geom
    #
    # def draw_line(self, start, end, **attrs):
    #     geom = Line(start, end)
    #     _add_attrs(geom, attrs)
    #     self.add_onetime(geom)
    #     return geom

    def get_array(self):
        # This function is similar to the return_rgb_array logic in render()
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]


class Sprite(object):
    def __init__(self, path, x=0, y=0):
        self.img = pyglet.image.load(path)
        # Set the anchor point of the sprite to its center for rotation around the center
        self.img.anchor_x = self.img.width // 2
        self.img.anchor_y = self.img.height // 2
        self.sprite = pyglet.sprite.Sprite(self.img, x=x, y=y)
    #
    def set_rotation(self, r):
        # Pyglet uses degrees for sprite rotation
        self.sprite.rotation = r # RAD2DEG

    def set_position(self, x, y):
        self.sprite.x = x #- self.sprite.image.width // 2
        self.sprite.y = y - self.sprite.image.height // 2

    def has_vertices(self):
        return False

    def draw(self):
        self.sprite.draw()


# Attr classes are largely removed in favor of direct property settings on shapes

class Geom(object):
    def __init__(self):
        # Default color (0,0,0) in shapes is handled differently, often via creation parameter
        self.transforms = []  # Placeholder for compatibility, but modern shapes handle this
        self.shape = None  # The actual pyglet.shapes object

    def add_transform(self, attr):
        self.transforms.append(attr)

    def set_color(self, r, g, b):
        if self.shape:
            # pyglet.shapes uses 0-255 uint8 tuples for color
            self.shape.color = (int(r * 255), int(g * 255), int(b * 255))



class Transform(object):

    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1.0, 1.0)):
        self.set_translation(*translation)
        self.set_rotation(rotation)  # in radians
        self.set_scale(*scale)
        self._matrix = Mat4()

    def _compute_matrix(self) -> Mat4:
        """Compute the local transformation matrix using Vec3 arguments."""
        sx, sy = self.scale
        tx, ty = self.translation
        rot = self.rotation

        # Note: Mat4 factory methods take Vec3 arguments
        m = Mat4().scale(Vec3(sx, sy, 1.0))
        m = m @ Mat4().rotate(rot * PI_LOCAL / 180.0, Vec3(0.0, 0.0, 1.0))
        m = m @ Mat4().translate(Vec3(tx, ty, 0.0))
        return m

    # setters for convenience
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)  # radians preferred
        # If you want to use degrees: self.rotation = radians(float(new))

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


# FilledPolygon and Line use pyglet.shapes
class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        # Convert list of vertices to a flat list for shapes.Polygon
        flat_vertices = [coord for point in v for coord in point]
        # numpy_array = np.array(v )
        self.shape = shapes.Polygon(*v, color=(128, 128, 128))  # Default color
        # (l,b), (l,t), (r,t), (r,b)
        x1, y1 = v[0]
        x2, y2 = v[3]
        self.shape.anchor_x = 0.5 * (x1 + x2)
        self.shape.anchor_y = 0.5 * (y1 + y2)

    def has_vertices(self):
        return True

    def render(self):
        self.shape.rotation = 0
        self.shape.position = (0, 0)
        for xform in reversed(self.transforms):
            if self.shape is not None and isinstance(xform,Transform):
                self.shape.rotation += -xform.rotation * 360/PI_LOCAL
                self.shape.position = tuple(map(lambda x, y: x + y, self.shape.position, xform.translation))
        self.shape.draw()

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        # Shapes.Line handles color and width internally
        self.shape = shapes.Line(start[0], start[1], end[0], end[1], thickness=1, color=(0, 0, 0))


    def render(self):
        for xform in reversed(self.transforms):
            if self.shape is not None and isinstance(xform,Transform):
                self.shape.rotation += xform.rotation
                self.shape.position = tuple(map(lambda x, y: x + y, self.shape.position, xform.translation))
        self.shape.draw()


