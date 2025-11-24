"""
Cartpole implementation adapted from the gym implementation
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
and based on the original by Rich Sutton et al.

Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import os
# import spaces
import sys
import time
import numpy as np

# Assuming your module is in a 'my_modules' subdirectory
sys.path.append(os.path.abspath('simulator'))
import spaces
class CartPoleEnv(object):

    def __init__(self, max_steps=200):

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.steps = 0
        self.max_steps = max_steps

        # Angle at which to fail the episode
        self.theta_threshold_radians = 36 * 2 * math.pi / 360
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self.viewer = None
        self.state = None

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        self.steps += 1
        done = done or self.steps >= self.max_steps

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.steps = 0
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        # scale = pix / meter
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        tire_diam = 64.0

        if self.viewer is None:
            # from gym.envs.classic_control import rendering
            import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.carttrans = rendering.Transform()
            # self.carttrans = rendering.Transform((screen_width / 2, screen_height / 2), 0, (1.0, 1.0))

            self.tire = rendering.Sprite("simulator/tire.png")
            # self.tire = rendering.Sprite("tire.png")
            self.viewer.add_sprite(self.tire)

            axleoffset = tire_diam/2.0
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.4, .6, .8)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_transform(self.poletrans)
            pole.add_transform(self.carttrans)
            self.viewer.add_geom(pole)

            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state

        # tire angle:
        # rev = dist / circ, gives number of revolutions
        # rev * 360 gives deg
        dist = scale * x[0]
        circ = tire_diam * np.pi
        revs = dist / circ
        deg = revs * 360.0

        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.tire.set_rotation(deg)
        self.tire.set_position(cartx, carty + tire_diam )
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


if __name__ == '__main__':
    env = CartPoleEnv()
    env.reset()
    frame = env.render(mode='rgb_array')
    step = 0.2
    for i in np.arange(-2,2+step,step):
        # env.tire.set_rotation(i * 3.14/180.0)
        # env.tire.set_position(350-i, 100)
        env.state = (i, 0, i*93, 0) # (x, x_dot, theta, theta_dot)
        # env.tire.draw()
        frame = env.render(mode='rgb_array')
        time.sleep(1)
    env.render(mode='rgb_array')
