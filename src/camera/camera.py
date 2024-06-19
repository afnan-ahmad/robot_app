import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import math
import time

# YOLO model setup
model = YOLO("yolov8n-seg.pt")
names = model.model.names

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2)
colorizer = rs.colorizer()

class AppState:
    def __init__(self):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

state = AppState()

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True
    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False
    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True
    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False
    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True
    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False
    if event == cv2.EVENT_MOUSEMOVE:
        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]
        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2
        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)
        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz
    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz
    state.prev_mouse = (x, y)

cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

def project(v):
    h, w = out.shape[:2]
    view_aspect = float(h)/w
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * (w*view_aspect, h) + (w/2.0, h/2.0)
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj

def view(v):
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

def pointcloud(out, verts, texcoords, color, painter=True):
    if painter:
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))
    if state.scale:
        proj *= 0.5**state.decimate
    h, w = out.shape[:2]
    j, i = proj.astype(np.uint32).T
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm
    cw, ch = color.shape[:2][::-1]
    if painter:
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)
    out[i[m], j[m]] = color[u[m], v[m]]

out = np.empty((480, 640, 3), dtype=np.uint8)

target_class = "cup"

while True:
    if not state.paused:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_image = cv2.resize(color_image, (640, 480))

        rotated_color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)

        results = model.predict(rotated_color_image)
        annotator = Annotator(rotated_color_image, line_width=2)

        target = (0,0)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            for mask, cls, bbox in zip(masks, clss, bboxes):
                if names[int(cls)] == target_class:
                    color = colors(int(cls), True)
                    points = np.array(mask, dtype=np.int32)

                    overlay = rotated_color_image.copy()
                    cv2.fillPoly(overlay, [points], color)
                    alpha = 0.5
                    cv2.addWeighted(overlay, alpha, rotated_color_image, 1 - alpha, 0, rotated_color_image)

                    annotator.seg_bbox(mask=mask, mask_color=color, det_label=names[int(cls)])

                    print(f"Class: {names[int(cls)]}, Bounding Box: {bbox}")

                    print(f"Points: {np.mean(points, axis=0)//2}")

                    target = np.mean(points, axis=0) // 2
                   
                    target = (int(target[0]), int(target[1]))

                    target2 = np.mean(points, axis=0)
                    target2 = (int(target2[0]), int(target2[1]))



        segmented_image = cv2.rotate(rotated_color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        segmented_image = cv2.resize(segmented_image, (640, 480))

        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())


        if state.color:
            mapped_frame, color_source = color_frame, segmented_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)

        pc.map_to(mapped_frame)

        v, t = points.get_vertices(), points.get_texture_coordinates()

        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
        

        verts_copy = np.copy(verts).reshape((240, 320, 3))
        texcoords_copy = np.copy(texcoords).reshape((320, 240, 2))

    
        cv2.imshow('Verts', verts_copy[:,:,2])
        print(verts_copy[240//2, 320//2])



    now = time.time()

    out.fill(0)

    pointcloud(out, verts, texcoords, color_source)

    dt = time.time() - now

    cv2.setWindowTitle(state.WIN_NAME, f"RealSense ({w}x{h}) {1.0/dt:.2f}FPS ({dt*1000:.2f}ms) {'PAUSED' if state.paused else ''}")

    depth_colormap = cv2.resize(depth_colormap, (640, 480))
    
    depth_colormap = cv2.rotate(depth_colormap, cv2.ROTATE_90_CLOCKWISE)
    cv2.circle(depth_colormap, target2, 2, (255, 0, 0), -1)
    cv2.imshow('Depth Image', depth_colormap)
    
    cv2.imshow('Target', target)
    cv2.imshow(state.WIN_NAME, out)
    print(f"Target: {target}")
    color_image = segmented_image.copy()
    cv2.imshow('Color Image', segmented_image)


    color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
    cv2.circle(color_image, target2, 2, (255, 0, 0), -1)
    cv2.imshow('Target', color_image)


    key = cv2.waitKey(1)

    if key == ord("r"):
        state.reset()
    if key == ord("p"):
        state.paused ^= True
    if key == ord("d"):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    if key == ord("z"):
        state.scale ^= True
    if key == ord("c"):
        state.color ^= True
    if key == ord("s"):
        cv2.imwrite('./out.png', out)
    if key == ord("e"):
        points.export_to_ply('./out.ply', mapped_frame)
    if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

pipeline.stop()