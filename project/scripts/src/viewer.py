from mujoco_py import MjViewer, functions
from mujoco_py.builder import cymj
from mujoco_py.generated import const
import mujoco_py.generated
from mujoco_py.utils import rec_copy, rec_assign
from multiprocessing import Process, Queue, freeze_support
import numpy as np
import glfw
import time
import copy
import imageio
from threading import Lock


class Viewer(MjViewer):
    """
    Override default viewer with our own controls.
    We want a toggle that switches the mouse between manipulation or camera
    """

    DEFAULT_VIDEO_CAPTURE_PATH = 'C:/dev/cs234/project/results/video/'
    DEFAULT_IMAGE_CAPTURE_PATH = 'C:/dev/cs234/project/results/img/'

    def __init__(self, sim,
            video_capture_path=DEFAULT_VIDEO_CAPTURE_PATH,
            image_capture_path=DEFAULT_IMAGE_CAPTURE_PATH):
        super().__init__(sim)

        self._video_path = video_capture_path + __name__ + "%07d.mp4"
        self._image_path = image_capture_path + __name__ + "%07d.png"
        self._selected_geomid = const.GEOM_NONE


    def raycast(self):
        """
        Refer to:
        https://github.com/deepmind/dm_control/blob/92f9913013face0468442cd0964d5973ea2089ea/dm_control/viewer/renderer.py#L468
        Casts a ray at cursor position into the scene.
        Returns geomid of first geom it intersects, or -1 if no geom. Also returns world coordinates
        of intersection
        """
        cursor_worldcoords = np.zeros(3, dtype=np.float64)

        # Window coordinates are (0,0) at top-left, (screenwidth, screenheight) at bottom-right
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        window_width, window_height = glfw.get_window_size(self.window)
        window_scaling = framebuffer_width * 1.0 / window_width
        cursor_screen_coords = np.zeros(2, dtype=np.int32)
        cursor_screen_coords[0], cursor_screen_coords[1] = glfw.get_cursor_pos(self.window)
        cursor_screen_coords[0] = cursor_screen_coords[0] * window_scaling
        cursor_screen_coords[1] = cursor_screen_coords[1] * window_scaling



    def cursor_raycast(self):
        """
        Equivalent to function.mjv_select
        http://mujoco.org/book/APIreference.html#mjv_select

        """

        cursor_worldcoords = np.zeros(3, dtype=np.float64)

        # Window coordinates are (0,0) at top-left, (screenwidth, screenheight) at bottom-right
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        window_width, window_height = glfw.get_window_size(self.window)
        window_scaling = framebuffer_width * 1.0 / window_width
        cursor_screen_coords = np.zeros(2, dtype=np.int32)
        cursor_screen_coords[0], cursor_screen_coords[1] = glfw.get_cursor_pos(self.window)
        cursor_screen_coords[0] = cursor_screen_coords[0] * window_scaling
        cursor_screen_coords[1] = cursor_screen_coords[1] * window_scaling

        # Viewport coords are (0,0) bottom-left, (1.0,1.0) at top-right
        cursor_viewport_coords = np.zeros(2, dtype=np.float32)
        cursor_viewport_coords[0] = cursor_screen_coords[0] / window_width
        cursor_viewport_coords[1] = 1.0 - (cursor_screen_coords[1] / window_height)

        rctx = self.sim.render_contexts[0] # We should only have one context active
        
        selected_body_id = functions.mjv_select( # pylint: ignore=no-member
            m = self.sim.model,
            d = self.sim.data,
            vopt = rctx.vopt,
            aspectratio = window_width / window_height,
            relx = cursor_viewport_coords[0],
            rely = cursor_viewport_coords[1],
            scn = rctx.scn,
            selpnt = cursor_worldcoords
        )
        if selected_body_id < 0:
            cursor_worldcoords = None
        return selected_body_id, cursor_worldcoords

    
    def move_body_to_cursor_ray(self):
        pass


    def _cursor_pos_callback(self, window, xpos, ypos):
        # xpos, ypos are standard screen coordinates with (0,0) at top left
        # print("x {0}, y {1}".format(xpos, ypos))

        if not (self._button_left_pressed or self._button_right_pressed):
            return

        # Shunt mouse controls to manipulate the model when CTRL is pressed
        ctrl_ispressed = (
            glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        # Determine whether to move, zoom or rotate view
        shift_ispressed = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            action = const.MOUSE_MOVE_H if shift_ispressed else const.MOUSE_MOVE_V
        elif self._button_left_pressed:
            action = const.MOUSE_ROTATE_H if shift_ispressed else const.MOUSE_ROTATE_V
        else:
            action = const.MOUSE_ZOOM
            
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)
        with self._gui_lock:
            if not ctrl_ispressed:
                self.move_camera(action, dx / height, dy / height)
            else:
                # todo. No idea how to feed in an mjvperturb
                """
                rctx = self.sim.render_contexts[0]
                pert 
                functions.mjv_moveperturb(self.model, self.model.data, action,
                    dx, dy, rctx.scn, )
                """
        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)


        

    def key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        elif key == glfw.KEY_TAB:  # Switches cameras.
            self.cam.fixedcamid += 1
            self.cam.type = const.CAMERA_FIXED
            if self.cam.fixedcamid >= self._ncam:
                self.cam.fixedcamid = -1
                self.cam.type = const.CAMERA_FREE
        elif key == glfw.KEY_H:  # hides all overlay.
            self._hide_overlay = not self._hide_overlay
        elif key == glfw.KEY_SPACE and self._paused is not None:  # stops simulation.
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        elif key == glfw.KEY_V or \
                (key == glfw.KEY_ESCAPE and self._record_video):  # Records video. Triggers with V or if in progress by ESC.
            self._record_video = not self._record_video
            if self._record_video:
                fps = (1 / self._time_per_render)
                self._video_process = Process(target=save_video,
                                  args=(self._video_queue, self._video_path % self._video_idx, fps))
                self._video_process.start()
            if not self._record_video:
                self._video_queue.put(None)
                self._video_process.join()
                self._video_idx += 1
            
        elif key == glfw.KEY_T:  # capture screenshot
            img = self._read_pixels_as_in_window()
            imageio.imwrite(self._image_path % self._image_idx, img)
            self._image_idx += 1
        elif key == glfw.KEY_I:  # drops in debugger.
            print('You can access the simulator by self.sim')
            import ipdb
            ipdb.set_trace()
        elif key == glfw.KEY_S:  # Slows down simulation.
            self._run_speed /= 2.0
        elif key == glfw.KEY_F:  # Speeds up simulation.
            self._run_speed *= 2.0
        elif key == glfw.KEY_C:  # Displays contact forces.
            vopt = self.vopt
            vopt.flags[10] = vopt.flags[11] = not vopt.flags[10]
        elif key == glfw.KEY_D:  # turn off / turn on rendering every frame.
            self._render_every_frame = not self._render_every_frame
        elif key == glfw.KEY_E:
            vopt = self.vopt
            vopt.frame = 1 - vopt.frame
        elif key == glfw.KEY_R:  # makes everything little bit transparent.
            self._transparent = not self._transparent
            if self._transparent:
                self.sim.model.geom_rgba[:, 3] /= 5.0
            else:
                self.sim.model.geom_rgba[:, 3] *= 5.0
        elif key == glfw.KEY_M:  # Shows / hides mocap bodies
            self._show_mocap = not self._show_mocap
            for body_idx1, val in enumerate(self.sim.model.body_mocapid):
                if val != -1:
                    for geom_idx, body_idx2 in enumerate(self.sim.model.geom_bodyid):
                        if body_idx1 == body_idx2:
                            if not self._show_mocap:
                                # Store transparency for later to show it.
                                self.sim.extras[
                                    geom_idx] = self.sim.model.geom_rgba[geom_idx, 3]
                                self.sim.model.geom_rgba[geom_idx, 3] = 0
                            else:
                                self.sim.model.geom_rgba[
                                    geom_idx, 3] = self.sim.extras[geom_idx]
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        super().key_callback(window, key, scancode, action, mods)


def save_video(queue, filename, fps):
    writer = imageio.get_writer(filename, fps=fps)
    while True:
        frame = queue.get()
        if frame is None:
            break
        writer.append_data(frame)
    writer.close()
