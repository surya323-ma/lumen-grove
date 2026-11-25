"""Lumen Grove - a living particle artwork.

Webcam + computer vision turn you into glowing particles inside a windswept
night grove. Run `python main.py` (or `python main.py --demo` with no camera).
"""
import argparse
import datetime
import os
import time

import cv2
import numpy as np

from scene import Scene, Renderer, PALETTES
from camera import CameraSegmenter, demo_mask, demo_hands
from hud import HUD, APP_TITLE

WINDOW = "Lumen Grove"


def parse():
    ap = argparse.ArgumentParser(description="Lumen Grove - living particle artwork")
    ap.add_argument('--camera', type=int, default=0, help='camera index')
    ap.add_argument('--width', type=int, default=840, help='window width in pixels')
    ap.add_argument('--art-height', type=int, default=600, help='window height in pixels')
    ap.add_argument('--cell', type=int, default=3, help='pixels per particle cell (smaller = finer, slower)')
    ap.add_argument('--fps', type=int, default=60, help='frame rate cap')
    ap.add_argument('--palette', type=int, default=1, choices=range(1, len(PALETTES) + 1),
                    help='start palette (1-%d)' % len(PALETTES))
    ap.add_argument('--seed', type=int, default=7, help='seed for the procedural tree and grass')
    ap.add_argument('--fullscreen', action='store_true')
    ap.add_argument('--no-panel', action='store_true', help='start with the camera preview hidden')
    ap.add_argument('--no-ui', action='store_true', help='start with the interface hidden (installation mode)')
    ap.add_argument('--invert', action='store_true', help='invert the segmentation mask')
    ap.add_argument('--demo', action='store_true', help='run without a camera')
    ap.add_argument('--snapshot', type=str, default='', help='render headless: out.png:FRAMES')
    return ap.parse_args()


def main():
    a = parse()
    cell = a.cell
    gw, gh = a.width // cell, a.art_height // cell
    scene = Scene(gw, gh, seed=a.seed)
    scene.set_palette(a.palette - 1)
    scene.cur_b, scene.cur_d, scene.cur_a = scene.tgt_b.copy(), scene.tgt_d.copy(), scene.tgt_a.copy()
    scene.cur_aur = scene.tgt_aur
    rend = Renderer(gw, gh, cell)
    W, H = rend.W, rend.H
    hud = HUD(W, H)
    hud.visible = not a.no_ui
    hud.pip = not a.no_panel

    cam = None
    if not a.demo and not a.snapshot:
        cam = CameraSegmenter(gw, gh, 320, 180, a.camera, a.invert)
        cam.start()
    demo = cam is None

    snap_path, snap_frames = '', 0
    if a.snapshot:
        snap_path, _, nf = a.snapshot.rpartition(':')
        snap_frames = int(nf)

    mouse = dict(x=0, y=0, moved=-99.0, down=False, click=None)
    start = time.perf_counter()

    def grid_pos(x, y):
        try:
            _, _, rw, rh = cv2.getWindowImageRect(WINDOW)
        except Exception:
            rw = rh = 0
        if rw <= 0 or rh <= 0:
            rw, rh = W, H
        return x * W / rw / cell, y * H / rh / cell

    def on_mouse(ev, x, y, flags, _):
        mouse['x'], mouse['y'] = x, y
        mouse['moved'] = time.perf_counter() - start
        if ev == cv2.EVENT_LBUTTONDOWN:
            mouse['down'] = True
            mouse['click'] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            mouse['down'] = False

    if not snap_path:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        cv2.setMouseCallback(WINDOW, on_mouse)
        if a.fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    full = a.fullscreen

    paused = False
    sim_t = 0.0
    fps = 0.0
    last = time.perf_counter()
    col = None
    panel = None
    person = False
    n = 0
    try:
        while True:
            tick = time.perf_counter()
            dt = min(tick - last, 0.1)
            last = tick
            clock = tick - start
            if snap_path:
                dt, clock = 1 / 30.0, 6.0 + n / 30.0
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.92 * fps + 0.08 / dt

            # mouse acts as a hand: stirs wind, attracts fireflies, click = burst
            if not snap_path:
                if clock - mouse['moved'] < 1.6:
                    gx, gy = grid_pos(mouse['x'], mouse['y'])
                    scene.set_pointer(gx, gy, 1.0 if mouse['down'] else 0.75)
                else:
                    scene.clear_pointer()
                if mouse['click']:
                    scene.burst(*grid_pos(*mouse['click']))
                    mouse['click'] = None

            if not paused or col is None:
                if not paused:
                    sim_t += dt
                if demo:
                    mask, luma, hands, panel = demo_mask(gw, gh, sim_t), None, demo_hands(gw, gh, sim_t), None
                else:
                    mask, luma, hands, panel = cam.latest()
                person = mask is not None and float(mask.mean()) > 0.004
                col = scene.frame(sim_t, mask, luma, hands)
            art = rend.draw(col)

            status = 'demo' if demo else ('live' if cam.active else 'searching')
            info = dict(status=status, fps=fps, palette=scene.palette_name, wind=scene.wind_name,
                        aurora=scene.aurora_on, person=person, glow=PALETTES[scene.palette]['glow'])
            img = hud.draw(art.copy(), clock, info, panel)
            n += 1

            if snap_path:
                if n >= snap_frames:
                    cv2.imwrite(snap_path, img)
                    break
                continue

            cv2.imshow(WINDOW, img)
            wait = max(1, int(1000.0 / a.fps - (time.perf_counter() - tick) * 1000.0))
            k = cv2.waitKey(wait) & 255
            if k != 255:
                hud.poke(clock)
            if k in (27, ord('q')):
                break
            elif k in (ord('1'), ord('2'), ord('3'), ord('4')):
                scene.set_palette(k - ord('1'))
                hud.toast("Palette: " + scene.palette_name, clock)
            elif k == ord('n'):
                scene.cycle_palette()
                hud.toast("Palette: " + scene.palette_name, clock)
            elif k == ord('a'):
                hud.toast("Aurora " + ("on" if scene.toggle_aurora() else "off"), clock)
            elif k == ord('w'):
                hud.toast("Wind: " + scene.cycle_wind(), clock)
            elif k == ord('v'):
                hud.pip = not hud.pip
                hud.toast("Camera view " + ("on" if hud.pip else "off"), clock)
            elif k == ord('h'):
                hud.visible = not hud.visible
                if hud.visible:
                    hud.poke(clock)
            elif k == ord(' '):
                paused = not paused
                hud.toast("Paused" if paused else "Resumed", clock)
            elif k == ord('d'):
                if cam is None:
                    hud.toast("No camera running (started with --demo)", clock)
                else:
                    demo = not demo
                    hud.toast("Demo mode on" if demo else "Live camera", clock)
            elif k == ord('s'):
                os.makedirs("snapshots", exist_ok=True)
                name = os.path.join("snapshots", "lumen-grove-%s.png" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                cv2.imwrite(name, art)
                hud.toast("Saved " + name, clock, 3.2)
            elif k == ord('f'):
                full = not full
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            elif k == ord('c') and cam is not None:
                cam.next_camera()
                hud.toast("Switching camera...", clock)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if cam is not None:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
