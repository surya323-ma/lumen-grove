"""Lumen Grove - on-screen interface: intro card, status, hints, toasts, camera preview."""
import math
import cv2
import numpy as np

APP_TITLE = "LUMEN GROVE"
APP_TAGLINE = "a living particle artwork"

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

HINTS = [("1-4", "palette"), ("A", "aurora"), ("W", "wind"), ("V", "camera view"), ("D", "demo"),
         ("Space", "pause"), ("S", "save"), ("F", "full"), ("H", "hide ui"), ("Q", "quit")]


def sstep(x, a, b):
    x = min(1.0, max(0.0, (x - a) / (b - a)))
    return x * x * (3 - 2 * x)


def scaled(color, a):
    return tuple(int(c * a) for c in color)


class HUD:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.u = max(1.0, W / 840.0)
        self.visible = True
        self.pip = True
        self.toasts = []
        self.hint_until = 11.0
        self.status = None
        self.status_since = 0.0
        self.last_person = 0.0
        # soft top/bottom shading so text stays readable over bright particles
        sh = np.ones((H, 1), np.float32)
        top, bot = int(95 * self.u), int(80 * self.u)
        y = np.arange(H, dtype=np.float32)
        sh[:, 0] = np.where(y < top, 0.35 + 0.65 * (y / top) ** 1.5, 1.0)
        sh[:, 0] = np.where(y > H - bot, np.minimum(sh[:, 0], 0.35 + 0.65 * ((H - y) / bot) ** 1.5), sh[:, 0])
        self.shade = np.dstack([np.repeat((sh * 255).astype(np.uint8), W, axis=1)] * 3)

    # ---------------------------------------------------------------- api
    def toast(self, msg, now, dur=2.4):
        self.toasts.append((msg, now, dur))

    def poke(self, now, secs=4.5):
        self.hint_until = max(self.hint_until, now + secs)

    # ---------------------------------------------------------------- text helpers
    def _width(self, s, scale, thick, spacing, font=FONT):
        return sum(cv2.getTextSize(ch, font, scale, thick)[0][0] + spacing for ch in s) - spacing

    def _text(self, ov, s, x, y, scale, color, alpha=1.0, thick=1, spacing=0.0, font=FONT):
        col = scaled(color, alpha)
        if spacing == 0:
            cv2.putText(ov, s, (int(x), int(y)), font, scale, col, thick, cv2.LINE_AA)
            return
        for ch in s:
            cv2.putText(ov, ch, (int(x), int(y)), font, scale, col, thick, cv2.LINE_AA)
            x += cv2.getTextSize(ch, font, scale, thick)[0][0] + spacing

    def _ctext(self, ov, s, cx, y, scale, color, alpha=1.0, thick=1, spacing=0.0, font=FONT):
        w = self._width(s, scale, thick, spacing, font)
        self._text(ov, s, cx - w / 2, y, scale, color, alpha, thick, spacing, font)

    # ---------------------------------------------------------------- main
    def draw(self, img, now, info, panel=None):
        u, W, H = self.u, self.W, self.H
        glow = info.get("glow", (255, 160, 90))
        status = info["status"]
        if status != self.status:
            self.status, self.status_since = status, now
        if info.get("person"):
            self.last_person = now

        if self.visible:
            img = cv2.multiply(img, self.shade, scale=1 / 255.0)
        ov = np.zeros_like(img)

        # intro card: fade up from black with the title in the middle
        dark = 1.0 - sstep(now, 0.7, 3.0)
        if dark > 0.001:
            img = cv2.convertScaleAbs(img, alpha=1.0 - dark * 0.96)
            ta = sstep(now, 0.15, 1.2) * (1 - sstep(now, 2.3, 3.2))
            if ta > 0.001:
                self._ctext(ov, APP_TITLE, W / 2, H / 2 - 4 * u, 1.45 * u, glow, ta, int(2 * u), 16 * u)
                self._ctext(ov, APP_TAGLINE.upper(), W / 2, H / 2 + 28 * u, 0.42 * u, (200, 190, 185),
                            ta * 0.8, 1, 5 * u, FONT_S)

        if self.visible:
            self._draw_chrome(ov, now, info, glow, status)
            if panel is not None and self.pip:
                self._pip(img, ov, panel, glow, now)

        self._draw_toast(ov, now, glow)
        return cv2.add(img, ov)

    def _draw_chrome(self, ov, now, info, glow, status):
        u, W, H = self.u, self.W, self.H
        # title
        a = sstep(now, 2.8, 4.0)
        if a > 0:
            self._text(ov, APP_TITLE, 22 * u, 36 * u, 0.62 * u, glow, 0.95 * a, 1, 5 * u)
            self._text(ov, APP_TAGLINE, 23 * u, 56 * u, 0.4 * u, (190, 180, 175), 0.65 * a, 1, 0, FONT_S)

        # status chip (top right)
        colors = {"live": (170, 255, 130), "demo": (70, 190, 255), "searching": (130, 120, 255)}
        labels = {"live": "LIVE", "demo": "DEMO", "searching": "SEARCHING CAMERA"}
        pulse = 0.65 + 0.35 * math.sin(now * 4.0)
        label = labels[status]
        sc = 0.46 * u
        right = W - 22 * u
        fps = "%d FPS" % round(info.get("fps", 0))
        wf = self._width(fps, 0.38 * u, 1, 0, FONT_S)
        self._text(ov, fps, right - wf, 34 * u, 0.38 * u, (190, 180, 175), 0.7, 1, 0, FONT_S)
        wl = self._width(label, sc, 1, 2 * u)
        lx = right - wf - 16 * u - wl
        self._text(ov, label, lx, 36 * u, sc, colors[status], 0.95, 1, 2 * u)
        cv2.circle(ov, (int(lx - 12 * u), int(31 * u)), max(3, int(4.5 * u)), scaled(colors[status], pulse), -1, cv2.LINE_AA)
        mode = "%s  /  wind %s  /  aurora %s" % (info["palette"], info["wind"], "on" if info["aurora"] else "off")
        wm = self._width(mode, 0.38 * u, 1, 0, FONT_S)
        self._text(ov, mode, right - wm, 56 * u, 0.38 * u, (190, 180, 175), 0.7, 1, 0, FONT_S)

        # guidance
        msg = None
        if status == "searching":
            msg = "LOOKING FOR YOUR CAMERA" + "." * (1 + int(now * 2) % 3)
        elif status == "live" and now - self.last_person > 2.5 and now > 4:
            msg = "STEP INTO THE FRAME  -  RAISE A HAND TO STIR THE GROVE"
        elif status == "demo" and now - self.status_since < 10 and now > 3.5:
            msg = "MOVE THE MOUSE TO STIR THE GROVE  -  CLICK FOR A BURST OF LIGHT"
        if msg:
            ga = (0.55 + 0.35 * math.sin(now * 2.2)) * sstep(now - self.status_since, 0.2, 0.9)
            sc2 = min(0.5 * u, 0.5 * u * (W - 60 * u) / max(1, self._width(msg, 0.5 * u, 1, 3 * u)))
            self._ctext(ov, msg, W / 2, 92 * u, sc2, glow, max(0.0, ga), 1, 3 * u)

        # hint bar
        ha = sstep(self.hint_until - now, 0, 1.0)
        if ha > 0.01:
            scale = 0.42 * u
            gap, kgap = 20 * u, 6 * u

            def total(s):
                return sum(self._width(k, s, 1, 0, FONT_S) + kgap + self._width(l, s, 1, 0, FONT_S) + gap
                           for k, l in HINTS) - gap
            avail = W - 40 * u
            if total(scale) > avail:
                scale *= avail / total(scale)
            x = (W - total(scale)) / 2
            y = H - 22 * u
            for k, l in HINTS:
                self._text(ov, k, x, y, scale, glow, 0.95 * ha, 1, 0, FONT_S)
                x += self._width(k, scale, 1, 0, FONT_S) + kgap
                self._text(ov, l, x, y, scale, (185, 178, 172), 0.8 * ha, 1, 0, FONT_S)
                x += self._width(l, scale, 1, 0, FONT_S) + gap

    def _pip(self, img, ov, panel, glow, now):
        u, W, H = self.u, self.W, self.H
        pw = int(W * 0.24)
        ph = int(pw * panel.shape[0] / panel.shape[1])
        x0, y0 = W - pw - int(22 * u), H - ph - int(58 * u)
        a = sstep(now, 3.5, 4.5)
        small = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_AREA)
        roi = img[y0:y0 + ph, x0:x0 + pw]
        img[y0:y0 + ph, x0:x0 + pw] = cv2.addWeighted(roi, 1 - 0.9 * a, small, 0.9 * a, 0)
        cv2.rectangle(ov, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), scaled(glow, 0.55 * a), 1, cv2.LINE_AA)
        self._text(ov, "WHAT THE CAMERA SEES", x0, y0 - 8 * u, 0.32 * u, glow, 0.7 * a, 1, 1.5 * u, FONT_S)

    def _draw_toast(self, ov, now, glow):
        self.toasts = [t for t in self.toasts if now - t[1] < t[2]]
        if not self.toasts:
            return
        msg, t0, dur = self.toasts[-1]
        a = sstep(now - t0, 0, 0.25) * (1 - sstep(now - t0, dur - 0.5, dur))
        self._ctext(ov, msg.upper(), self.W / 2, self.H - 62 * self.u, 0.55 * self.u, glow, a, 1, 3 * self.u)
