"""Lumen Grove - the procedural night scene.

Everything is drawn on a small grid of "cells" (one cell = one glowing dot).
Layers:  stars, aurora, moon, meteors, tree, grass, mist, fireflies, sparks,
the live person silhouette and the hand skeletons.
"""
import math
import numpy as np
import cv2


def _stops(*pairs):
    return [(p[0], p[1]) for p in pairs]


# Each palette: base ramp, accent ramp for the "alive" particles (fireflies,
# sparks, embers) and a secondary tint that is patch-mixed over the base ramp.
PALETTES = [
    dict(name="Electric", glow=(255, 160, 90), aurora=0.55,
         base=_stops((0.0, (0, 0, 0)), (0.12, (6, 6, 60)), (0.35, (25, 30, 190)),
                     (0.6, (70, 80, 255)), (0.82, (150, 165, 255)), (1.0, (235, 240, 255))),
         tint=_stops((0.0, (0, 0, 0)), (0.12, (30, 4, 70)), (0.35, (95, 25, 205)),
                     (0.6, (140, 70, 255)), (0.82, (195, 160, 255)), (1.0, (240, 230, 255))),
         accent=_stops((0.0, (0, 0, 0)), (0.2, (0, 60, 90)), (0.5, (0, 205, 255)),
                       (0.8, (140, 245, 255)), (1.0, (235, 255, 255)))),
    dict(name="Aurora", glow=(170, 255, 120), aurora=1.5,
         base=_stops((0.0, (0, 0, 0)), (0.12, (0, 40, 40)), (0.35, (0, 150, 120)),
                     (0.6, (40, 230, 160)), (0.82, (160, 255, 210)), (1.0, (235, 255, 245))),
         tint=_stops((0.0, (0, 0, 0)), (0.12, (40, 10, 60)), (0.35, (110, 40, 170)),
                     (0.6, (170, 90, 240)), (0.82, (215, 170, 255)), (1.0, (245, 235, 255))),
         accent=_stops((0.0, (0, 0, 0)), (0.2, (90, 10, 60)), (0.5, (255, 70, 170)),
                       (0.8, (255, 160, 215)), (1.0, (255, 235, 245)))),
    dict(name="Ember", glow=(60, 150, 255), aurora=0.5,
         base=_stops((0.0, (0, 0, 0)), (0.12, (60, 8, 0)), (0.35, (190, 40, 10)),
                     (0.6, (255, 100, 30)), (0.82, (255, 185, 110)), (1.0, (255, 240, 225))),
         tint=_stops((0.0, (0, 0, 0)), (0.12, (70, 0, 40)), (0.35, (190, 20, 100)),
                     (0.6, (255, 60, 150)), (0.82, (255, 150, 200)), (1.0, (255, 235, 245))),
         accent=_stops((0.0, (0, 0, 0)), (0.2, (80, 40, 0)), (0.5, (255, 200, 40)),
                       (0.8, (255, 235, 140)), (1.0, (255, 252, 215)))),
    dict(name="Frost", glow=(255, 225, 190), aurora=0.7,
         base=_stops((0.0, (0, 0, 0)), (0.12, (20, 30, 50)), (0.35, (80, 120, 170)),
                     (0.6, (150, 200, 235)), (0.82, (210, 235, 250)), (1.0, (250, 252, 255))),
         tint=_stops((0.0, (0, 0, 0)), (0.12, (35, 25, 60)), (0.35, (110, 90, 175)),
                     (0.6, (170, 150, 235)), (0.82, (215, 205, 250)), (1.0, (250, 247, 255))),
         accent=_stops((0.0, (0, 0, 0)), (0.2, (60, 50, 10)), (0.5, (255, 215, 90)),
                       (0.8, (255, 240, 170)), (1.0, (255, 252, 225)))),
]

WIND_LEVELS = [("calm", 0.45), ("breeze", 1.0), ("storm", 1.9)]

HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
             (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
             (18, 19), (19, 20), (0, 17)]


def _make_lut(stops):
    v = np.linspace(0, 1, 256)
    xs = [s[0] for s in stops]
    out = np.zeros((256, 3), np.float32)
    for c in range(3):
        out[:, c] = np.interp(v, xs, [s[1][c] for s in stops])
    return out[:, ::-1].copy()  # RGB -> BGR


def smooth_noise(rng, h, w, ch, cw):
    small = rng.random((ch, cw)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def wind(x, t):
    gust = 0.75 + 0.25 * math.sin(t * 0.23)
    return gust * (0.65 * np.sin(x * 0.045 - t * 1.15) + 0.35 * np.sin(x * 0.12 - t * 2.4 + 1.3)) \
        + 0.25 * math.sin(t * 0.7)


class Scene:

    def __init__(self, gw, gh, seed=7):
        self.gw, self.gh = gw, gh
        self.rng = np.random.default_rng(seed)
        self.ground = int(gh * 0.72)
        self.k = gh / 144.0
        self.dens = max(0.5, gw * gh / 32400.0)

        # ---- look & feel state ------------------------------------------------
        self.palette = 0
        self.aurora_on = True
        self.wind_idx = 1
        self.wind_gain = 1.0
        self._luts = [_make_lut(p["base"]) for p in PALETTES]
        self._luts_t = [_make_lut(p["tint"]) for p in PALETTES]
        self._luts_a = [_make_lut(p["accent"]) for p in PALETTES]
        self._set_targets()
        self.cur_b = self.tgt_b.copy()
        self.cur_d = self.tgt_d.copy()
        self.cur_a = self.tgt_a.copy()
        self.cur_aur = self.tgt_aur
        self.aur_mix = 1.0

        self.vmap = np.clip(smooth_noise(self.rng, gh, gw, 4, 7) * 1.2 - 0.55, 0, 0.3)[..., None]
        self.m_ema = np.zeros((gh, gw), np.float32)
        self.XX = np.arange(gw, dtype=np.float32)[None, :]
        self.YY = np.arange(gh, dtype=np.float32)[:, None]
        yy = self.YY
        self.sitfade = self._smooth(np.clip((int(gh * 0.80) - yy) / (0.07 * gh), 0, 1))
        self.rowfade = 1.0 - 0.55 * np.clip((yy - self.ground) / max(1, gh - self.ground), 0, 1)

        # hands / pointer
        self.hp = [0.0, 0.0]
        self.hpts = [None, None]
        self._prev_tips = [None, None]
        self.ptr = None
        self._prev_ptr = None
        self._bursts = []
        self._last_t = None

        # particle pools
        self.sx = self.sy = self.svx = self.svy = self.sl = self.sg = self.sd = np.zeros(0, np.float32)
        self.sacc = np.zeros(0, bool)
        self.met = None

        self._build_stars()
        self._build_moon()
        self._build_tree()
        self._build_grass()
        self._build_mist()
        self._build_fireflies()

    # ------------------------------------------------------------------ controls
    def _set_targets(self):
        i = self.palette
        self.tgt_b = self._luts[i]
        self.tgt_d = self._luts_t[i] - self._luts[i]
        self.tgt_a = self._luts_a[i]
        self.tgt_aur = PALETTES[i]["aurora"]

    def set_palette(self, i):
        self.palette = i % len(PALETTES)
        self._set_targets()

    def cycle_palette(self):
        self.set_palette(self.palette + 1)

    def toggle_aurora(self):
        self.aurora_on = not self.aurora_on
        return self.aurora_on

    def cycle_wind(self):
        self.wind_idx = (self.wind_idx + 1) % len(WIND_LEVELS)
        self.wind_gain = WIND_LEVELS[self.wind_idx][1]
        return WIND_LEVELS[self.wind_idx][0]

    @property
    def palette_name(self):
        return PALETTES[self.palette]["name"]

    @property
    def wind_name(self):
        return WIND_LEVELS[self.wind_idx][0]

    def set_pointer(self, x, y, s=0.8):
        self.ptr = (float(x), float(y), float(s))

    def clear_pointer(self):
        self.ptr = None
        self._prev_ptr = None

    def burst(self, x, y):
        self._bursts.append((float(x), float(y)))

    def _ease_palette(self):
        self.cur_b += (self.tgt_b - self.cur_b) * 0.12
        self.cur_d += (self.tgt_d - self.cur_d) * 0.12
        self.cur_a += (self.tgt_a - self.cur_a) * 0.12
        self.cur_aur += (self.tgt_aur - self.cur_aur) * 0.1

    # ------------------------------------------------------------------ builders
    def _build_stars(self):
        r, gw, g = self.rng, self.gw, self.ground
        n = int(gw * g * 0.012)
        self.s_x = r.integers(0, gw, n).astype(np.float32)
        self.s_y = (r.random(n) ** 1.2 * (g - 8)).astype(np.float32)
        self.s_b = (0.25 + 0.6 * r.random(n)).astype(np.float32)
        self.s_ph = r.uniform(0, 6.28, n).astype(np.float32)
        self.s_sp = r.uniform(0.6, 3.0, n).astype(np.float32)

    def _build_moon(self):
        gw, gh = self.gw, self.gh
        mx, my, R = 0.12 * gw, 0.15 * gh, 0.055 * gh
        d = np.hypot(self.XX - mx, self.YY - my)
        disc = np.clip((R - d) / 1.3, 0, 1)
        halo = np.exp(-(d / (R * 2.4)) ** 2) * 0.42 * (1 - disc)
        crater = smooth_noise(self.rng, gh, gw, 9, 14) * 0.3
        self.moon = np.clip(disc * (0.92 - crater) + halo, 0, 1).astype(np.float32)

    def _build_mist(self):
        gw, gh, g = self.gw, self.gh, self.ground
        self.mist_noise = smooth_noise(self.rng, gh, gw * 2, 4, 12)
        self.mist_band = np.exp(-((self.YY - (g + 2)) / (0.055 * gh)) ** 2).astype(np.float32)

    def _build_fireflies(self):
        r, gw, gh = self.rng, self.gw, self.gh
        n = int(46 * self.dens ** 0.7)
        self.fx = r.uniform(0, gw, n).astype(np.float32)
        self.fy = r.uniform(0.4 * gh, 0.95 * gh, n).astype(np.float32)
        self.fa = r.uniform(0, 6.28, n).astype(np.float32)
        self.fs = (r.uniform(4, 10, n) * self.k).astype(np.float32)
        self.fph = r.uniform(0, 6.28, n).astype(np.float32)
        self.fsp = r.uniform(0.8, 2.2, n).astype(np.float32)

    def _branch(self, x, y, ang, length, depth, width, maxd, acc, leaves):
        r = self.rng
        n = max(2, int(length / 0.6))
        t = np.linspace(0, 1, n)
        bend = r.uniform(-0.18, 0.18) * length
        dx, dy = math.sin(ang), -math.cos(ang)
        px, py = -dy, dx
        cx = x + dx * length * t + px * bend * np.sin(t * math.pi)
        cy = y + dy * length * t + py * bend * np.sin(t * math.pi)
        half = max(0, int(width / 2))
        for k in range(-half, half + 1):
            acc.append(np.stack([cx + px * k, cy + py * k, 0.55 + 0.4 * r.random(n)], 1))
        ex, ey = cx[-1], cy[-1]
        if depth >= 2 and r.random() < 0.5:
            leaves.append((ex, ey))
        if depth >= maxd or length < 5:
            leaves.append((ex, ey))
            return
        for _ in range(int(r.choice([2, 3], p=[0.55, 0.45]))):
            na = float(np.clip(ang + r.uniform(-0.95, 0.95), -1.9, 1.9))
            self._branch(ex, ey, na, length * r.uniform(0.62, 0.8), depth + 1, width * 0.72, maxd, acc, leaves)

    def _fronds(self, cx, cy, rx, ry, n, parts, lo=4, hi=10):
        r, k = self.rng, self.k
        px = r.normal(cx, rx, n)
        py = r.normal(cy, ry, n)
        ang = np.pi / 2 + r.normal(0, 0.95, n)
        L = r.uniform(lo, hi, n) * k
        t = np.linspace(0, 1, int(hi * k / 0.7) + 1)[None, :]
        xs = px[:, None] + np.cos(ang)[:, None] * L[:, None] * t
        ys = py[:, None] + np.sin(ang)[:, None] * L[:, None] * t + 0.12 * L[:, None] * t * t
        b = 0.35 + 0.6 * r.random(n)[:, None] * (1 - 0.45 * t)
        parts.append(np.stack([xs.ravel(), ys.ravel(), b.ravel()], 1))

    def _build_tree(self):
        r, gw, gh, g = self.rng, self.gw, self.gh, self.ground
        tx, th = 0.84 * gw, 0.70 * gh
        acc, leaves, parts = [], [], []
        top = g - th
        for row in range(int(top), g + 3):
            frac = max(0.0, (g - row) / th)
            wd = 0.022 * gw * (1 - 0.45 * frac) + 3 * max(0, 1 - frac * 8) + 1
            cxm = tx + math.sin(frac * 3.0) * 1.5
            xs = np.arange(cxm - wd / 2, cxm + wd / 2 + 0.01, 1.0)
            b = 0.5 + 0.4 * r.random(len(xs))
            b[0] = b[-1] = 0.9
            parts.append(np.stack([xs, np.full_like(xs, row), b], 1))
        specs = [(1.0, -1.3, .30), (0.97, -0.8, .34), (0.95, -0.4, .30), (1.0, 0.5, .26), (0.9, 1.1, .2),
                 (0.82, -1.2, .30), (0.72, 1.25, .2), (0.66, -1.1, .26), (0.55, 1.0, .16), (0.5, -1.2, .2)]
        for fr, ang, l in specs:
            self._branch(tx, g - fr * th, ang, l * th, 0, 2, 4, acc, leaves)
        parts.extend(acc)
        nb = sum(len(p) for p in parts)
        self._fronds(0.50 * gw, 0.17 * gh, 0.20 * gw, 0.10 * gh, int(1400 * self.dens ** 0.35), parts)
        self._fronds(0.84 * gw, 0.12 * gh, 0.10 * gw, 0.08 * gh, int(250 * self.dens ** 0.35), parts)
        for lx, ly in leaves:
            self._fronds(lx, ly, 2.5 * self.k, 2.5 * self.k, 14, parts, 3, 7)
        P = np.concatenate(parts).astype(np.float32)
        leaf = np.zeros(len(P), np.float32)
        leaf[nb:] = 1.0
        ok = (P[:, 0] > -3) & (P[:, 0] < gw + 3) & (P[:, 1] > -3) & (P[:, 1] < g + 4)
        P, leaf = P[ok], leaf[ok]
        self.t_x, self.t_y, self.t_b, self.t_leaf = P[:, 0], P[:, 1], P[:, 2], leaf
        height = np.clip((g - P[:, 1]) / g, 0, 1)
        self.t_w = 1.9 * height ** 1.8 + 0.5 * leaf
        self.t_ph = r.uniform(0, 6.28, len(P)).astype(np.float32)
        self.t_sp = r.uniform(1.5, 4.0, len(P)).astype(np.float32)
        self.leaf_idx = np.flatnonzero(leaf > 0.5)

    def _build_grass(self):
        r, gw, gh, g = self.rng, self.gw, self.gh, self.ground
        n = int(gw * (gh - g) * 0.55)
        bx = r.random(n) * gw
        by = g - 4 + r.random(n) ** 0.85 * (gh - g + 4)
        d = np.clip((by - (g - 4)) / (gh - g + 4), 0, 1)
        h = (4 + r.random(n) * 10) * (0.7 + 0.9 * d) * self.k
        h *= np.where(r.random(n) < 0.08, 1.6, 1.0)
        h = np.maximum(2, h.astype(np.int32))
        noise = smooth_noise(r, gh, gw, 7, 14)
        keep = r.random(n) < np.clip(
            0.18 + noise[np.clip(by.astype(int), 0, gh - 1), np.clip(bx.astype(int), 0, gw - 1)] * 1.1, 0, 1)
        bx, by, d, h = bx[keep], by[keep], d[keep], h[keep]
        n = len(bx)
        bl = np.repeat(np.arange(n), h)
        k = np.arange(len(bl)) - np.repeat(np.cumsum(h) - h, h)
        frac = (k / h[bl]).astype(np.float32)
        base = (0.16 + 0.3 * d) * (0.55 + 0.45 * r.random(n))
        self.g_x = bx[bl].astype(np.float32)
        self.g_y = (by[bl] - k).astype(np.float32)
        self.g_frac = frac
        self.g_b = np.clip(base[bl] + 0.34 * frac ** 1.5, 0, 0.72).astype(np.float32)
        amp = (0.8 + 1.4 * r.random(n)) * (0.6 + 0.6 * d)
        self.g_amp = amp[bl].astype(np.float32)
        self.g_ph = r.uniform(0, 6.28, n)[bl].astype(np.float32)
        self.g_fr = d[bl] > 0.55
        self.g_sp = r.uniform(1.0, 3.2, n)[bl].astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def _splat(self, F, x, y, v):
        gw, gh = self.gw, self.gh
        xi = np.rint(x).astype(np.int32)
        yi = np.rint(y).astype(np.int32)
        ok = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)
        idx = yi[ok] * gw + xi[ok]
        flat = F.reshape(-1)
        flat[idx] = np.maximum(flat[idx], v[ok])

    def _glow_splat(self, F, x, y, v):
        """A point with a tiny plus-shaped halo (used for fireflies and sparks)."""
        self._splat(F, x, y, v)
        h = v * 0.42
        self._splat(F, x + 1, y, h)
        self._splat(F, x - 1, y, h)
        self._splat(F, x, y + 1, h)
        self._splat(F, x, y - 1, h)

    @staticmethod
    def _smooth(x):
        return x * x * (3 - 2 * x)

    def _emit(self, x, y, vx, vy, n, grav=20.0, decay=(0.8, 1.6), accent=True):
        if n <= 0:
            return
        r = self.rng
        n = int(n)
        x = np.broadcast_to(np.asarray(x, np.float32), (n,)).copy()
        y = np.broadcast_to(np.asarray(y, np.float32), (n,)).copy()
        vx = np.broadcast_to(np.asarray(vx, np.float32), (n,)).copy()
        vy = np.broadcast_to(np.asarray(vy, np.float32), (n,)).copy()
        self.sx = np.concatenate([self.sx, x])
        self.sy = np.concatenate([self.sy, y])
        self.svx = np.concatenate([self.svx, vx])
        self.svy = np.concatenate([self.svy, vy])
        self.sl = np.concatenate([self.sl, np.ones(n, np.float32)])
        self.sg = np.concatenate([self.sg, np.full(n, grav * self.k, np.float32)])
        self.sd = np.concatenate([self.sd, r.uniform(decay[0], decay[1], n).astype(np.float32)])
        self.sacc = np.concatenate([self.sacc, np.full(n, accent, bool)])
        if len(self.sx) > 1800:
            for name in ("sx", "sy", "svx", "svy", "sl", "sg", "sd", "sacc"):
                setattr(self, name, getattr(self, name)[-1800:])

    def _update_sparks(self, t, dt):
        if len(self.sx) == 0:
            return
        gw, gh = self.gw, self.gh
        self.svy = self.svy + self.sg * dt
        self.svx = self.svx * (1 - 0.8 * dt)
        self.sx = self.sx + self.svx * dt
        self.sy = self.sy + self.svy * dt
        self.sl = self.sl - self.sd * dt
        alive = (self.sl > 0) & (self.sx > -2) & (self.sx < gw + 2) & (self.sy > -2) & (self.sy < gh + 2)
        if not alive.all():
            for name in ("sx", "sy", "svx", "svy", "sl", "sg", "sd", "sacc"):
                setattr(self, name, getattr(self, name)[alive])

    def _update_hands(self, hands):
        for i in range(2):
            if hands and i < len(hands):
                pts, pr = hands[i]
                self.hpts[i] = pts if self.hpts[i] is None else 0.5 * self.hpts[i] + 0.5 * pts
                self.hp[i] = self.hp[i] * 0.75 + pr * 0.25
            else:
                self.hp[i] *= 0.85

    def _influences(self):
        """Things that stir the world: each hand and the mouse pointer."""
        out = []
        for i in range(2):
            s = float(np.clip((self.hp[i] - 0.12) / 0.88, 0, 1))
            pts = self.hpts[i]
            if s <= 0 or pts is None:
                continue
            c = pts[[0, 5, 9, 13, 17]].mean(0)
            R = float(np.hypot(*pts[9] - pts[0])) * 1.8 + 6
            out.append((float(c[0]), float(c[1]), R, s))
        if self.ptr is not None:
            px, py, ps = self.ptr
            out.append((px, py, 9 * self.k + 5, ps))
        return out

    def _emit_from_inputs(self, dt):
        r, k = self.rng, self.k
        # fingertips
        for i in range(2):
            s = float(np.clip((self.hp[i] - 0.12) / 0.88, 0, 1))
            pts = self.hpts[i]
            if s <= 0 or pts is None:
                self._prev_tips[i] = None
                continue
            tips = pts[[4, 8, 12, 16, 20]].copy()
            prev = self._prev_tips[i]
            self._prev_tips[i] = tips
            if prev is None:
                continue
            vel = (tips - prev) / max(dt, 1e-3)
            sp = np.hypot(vel[:, 0], vel[:, 1])
            for j in range(5):
                n = int(np.clip(sp[j] / (20 * k), 0, 4)) + (1 if r.random() < 0.18 * s else 0)
                if n:
                    self._emit(tips[j, 0] + r.normal(0, 0.4, n), tips[j, 1] + r.normal(0, 0.4, n),
                               vel[j, 0] * 0.22 + r.normal(0, 6 * k, n),
                               vel[j, 1] * 0.22 + r.normal(0, 6 * k, n) - 4 * k, n)
        # mouse pointer
        if self.ptr is not None:
            px, py, ps = self.ptr
            if self._prev_ptr is not None:
                dx, dy = px - self._prev_ptr[0], py - self._prev_ptr[1]
                sp = math.hypot(dx, dy) / max(dt, 1e-3)
                n = int(np.clip(sp / (14 * k), 0, 5)) + (1 if r.random() < 0.25 * ps else 0)
                if n:
                    self._emit(px + r.normal(0, 0.5, n), py + r.normal(0, 0.5, n),
                               dx / max(dt, 1e-3) * 0.15 + r.normal(0, 7 * k, n),
                               dy / max(dt, 1e-3) * 0.15 + r.normal(0, 7 * k, n) - 4 * k, n)
            self._prev_ptr = (px, py)
        # click bursts
        while self._bursts:
            bx, by = self._bursts.pop()
            n = 70
            a = r.uniform(0, 6.2832, n)
            sp = r.uniform(10, 46, n) * k
            self._emit(bx, by, np.cos(a) * sp, np.sin(a) * sp, n, grav=12, decay=(0.7, 1.4))

    # ------------------------------------------------------------------ layers
    def _aurora(self, t, gain):
        g, x, y = self.ground, self.XX, self.YY
        yc = g * 0.30 + 0.10 * g * np.sin(x * 0.035 + t * 0.28) + 0.05 * g * np.sin(x * 0.09 - t * 0.5)
        wid = g * (0.07 + 0.03 * np.sin(x * 0.05 + t * 0.2))
        d = (y - yc) / wid
        band = np.exp(-d * d)
        curtain = 0.5 + 0.5 * np.sin(x * 0.11 - t * 0.7 + 2.0 * np.sin(x * 0.03 + t * 0.15))
        inten = band * (0.25 + 0.75 * curtain) * 0.55 * gain
        inten = inten * (y < g * 0.78)
        lit = self.rng.random(inten.shape, dtype=np.float32) < inten
        return np.where(lit, np.clip(inten * 0.8 + 0.12, 0, 0.55), 0).astype(np.float32)

    def _update_meteor(self, E, dt):
        r, k = self.rng, self.k
        if self.met is None:
            if r.random() < 0.10 * dt:
                sgn = 1.0 if r.random() < 0.5 else -1.0
                ang = r.uniform(0.3, 0.65)
                sp = r.uniform(75, 115) * k
                x = r.uniform(0.15, 0.45) * self.gw if sgn > 0 else r.uniform(0.55, 0.85) * self.gw
                self.met = [x, r.uniform(0.02, 0.18) * self.gh, sgn * math.cos(ang) * sp, math.sin(ang) * sp, 1.0]
            return
        m = self.met
        m[0] += m[2] * dt
        m[1] += m[3] * dt
        m[4] -= dt * 0.85
        if m[4] <= 0 or m[1] > self.ground * 0.9 or m[0] < -5 or m[0] > self.gw + 5:
            self.met = None
            return
        i = np.arange(18, dtype=np.float32)
        tx = m[0] - m[2] * i * 0.011
        ty = m[1] - m[3] * i * 0.011
        v = ((1 - i / 18) ** 1.4 * min(1.0, m[4] * 2.5)).astype(np.float32)
        self._splat(E, tx, ty, v)

    def _hand_layer(self, t):
        gw, gh, r = self.gw, self.gh, self.rng
        H = np.zeros((gh, gw), np.float32)
        for i in range(2):
            s = float(np.clip((self.hp[i] - 0.12) / 0.88, 0, 1))
            pts = self.hpts[i]
            if s <= 0 or pts is None:
                continue
            xs, ys, vs = [], [], []
            for a, b in HAND_CONN:
                d = float(np.hypot(*pts[b] - pts[a]))
                n = int(d * 1.6) + 2
                tt = np.linspace(0, 1, n)
                for _ in range(int(2 + 4 * s)):
                    jit = 0.25 + 0.6 * s
                    xs.append(pts[a, 0] + (pts[b, 0] - pts[a, 0]) * tt + r.normal(0, jit, n))
                    ys.append(pts[a, 1] + (pts[b, 1] - pts[a, 1]) * tt + r.normal(0, jit, n))
                    vs.append(0.6 + 0.4 * s * r.random(n))
            c = pts[[0, 5, 9, 13, 17]].mean(0)
            size = float(np.hypot(*pts[9] - pts[0]))
            n = int(150 * s) + 30
            xs.append(c[0] + r.normal(0, size * 0.45, n))
            ys.append(c[1] + r.normal(0, size * 0.45, n))
            vs.append(0.5 + 0.4 * r.random(n))
            for k in (4, 8, 12, 16, 20):
                n = int(30 * s) + 8
                rad = 0.6 + 1.2 * s
                xs.append(pts[k, 0] + r.normal(0, rad, n))
                ys.append(pts[k, 1] + r.normal(0, rad, n))
                vs.append(np.full(n, 0.97))
            n = int(60 * s) + 10
            ang = r.uniform(0, 6.28, n)
            rr = size * (1.5 + 0.2 * np.sin(t * 3 + ang * 3))
            xs.append(c[0] + np.cos(ang) * rr)
            ys.append(c[1] + np.sin(ang) * rr)
            vs.append(np.full(n, 0.45 * s + 0.1))
            x = np.concatenate(xs).astype(np.float32)
            y = np.concatenate(ys).astype(np.float32)
            v = np.concatenate(vs).astype(np.float32)
            keep = r.random(len(x)) < 0.35 + 0.65 * s
            self._splat(H, x[keep], y[keep], v[keep])
        return H

    def _update_fireflies(self, t, dt, infl, X):
        r, k = self.rng, self.k
        n = len(self.fx)
        self.fa += r.normal(0, 2.4, n).astype(np.float32) * math.sqrt(dt)
        vx = np.cos(self.fa) * self.fs + wind(self.fx, t) * 5 * k * self.wind_gain
        vy = np.sin(self.fa) * self.fs * 0.6
        for cx, cy, R, s in infl:
            dx, dy = cx - self.fx, cy - self.fy
            d = np.sqrt(dx * dx + dy * dy) + 1e-3
            w = s * np.exp(-d / (60 * k))
            radial = np.clip((d - 14 * k) / (14 * k), -1, 1) * w * 48 * k
            tang = w * 36 * k
            vx = vx + dx / d * radial - dy / d * tang
            vy = vy + dy / d * radial + dx / d * tang
        self.fx = (self.fx + vx * dt) % self.gw
        self.fy = self.fy + vy * dt
        bad = (self.fy < 0.32 * self.gh) | (self.fy > self.gh - 3)
        self.fa[bad] += math.pi
        self.fy = np.clip(self.fy, 0.32 * self.gh, self.gh - 3)
        blink = (0.5 + 0.5 * np.sin(t * self.fsp + self.fph)) ** 2
        self._glow_splat(X, self.fx, self.fy, (0.25 + 0.72 * blink).astype(np.float32))

    def _drip_leaves(self, t):
        """Occasionally a canopy particle lets go and drifts down on the wind."""
        r, k = self.rng, self.k
        if len(self.leaf_idx) and r.random() < 0.30:
            i = int(r.choice(self.leaf_idx))
            self._emit(self.t_x[i], self.t_y[i], 3 * k * self.wind_gain + r.normal(0, 3 * k),
                       r.uniform(1, 5) * k, 1, grav=2.5, decay=(0.25, 0.5), accent=False)

    # ------------------------------------------------------------------ main frame
    def frame(self, t, mask=None, luma=None, hands=None):
        gw, gh, g, r, k = self.gw, self.gh, self.ground, self.rng, self.k
        dt = float(np.clip(t - self._last_t, 1e-3, 0.1)) if self._last_t is not None else 1 / 30.0
        self._last_t = t
        self._ease_palette()
        E = np.zeros((gh, gw), np.float32)
        X = np.zeros((gh, gw), np.float32)   # accent-coloured "alive" particles
        self._update_hands(hands)
        infl = self._influences()
        wg = self.wind_gain

        # sky
        tw = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(t * self.s_sp + self.s_ph)) ** 3
        self._splat(E, self.s_x, self.s_y, self.s_b * tw)
        self.aur_mix += ((1.0 if self.aurora_on else 0.0) - self.aur_mix) * 0.08
        if self.aur_mix * self.cur_aur > 0.03:
            E = np.maximum(E, self._aurora(t, self.cur_aur * self.aur_mix))
        glow = 0.94 + 0.06 * math.sin(t * 0.6)
        moon_lit = r.random((gh, gw), dtype=np.float32) < np.clip(self.moon * 1.35, 0, 1)
        E = np.maximum(E, np.where(moon_lit, self.moon * glow, 0).astype(np.float32))
        self._update_meteor(E, dt)

        # tree + grass with wind and body/hand/mouse interaction
        wt = wind(self.t_x, t) * wg
        dx = wt * 2.2 * k * self.t_w + self.t_leaf * 0.7 * k * np.sin(t * self.t_sp + self.t_ph)
        dy = self.t_leaf * 0.5 * np.cos(t * self.t_sp * 0.8 + self.t_ph)
        shim = 0.72 + 0.28 * np.sin(t * self.t_sp * 1.7 + self.t_ph * 2)
        wgr = wind(self.g_x, t) * wg + 0.5 * np.sin(t * self.g_sp + self.g_ph)
        gdx = wgr * self.g_amp * 1.6 * k * self.g_frac ** 1.3
        for cx, cy, R, s in infl:
            ddx = self.g_x - cx
            gdx += np.sign(ddx) * np.exp(-(ddx * ddx + (self.g_y - cy) ** 2) / (2 * R * R)) * s * 5 * self.g_frac
            tdx = self.t_x - cx
            dx += np.sign(tdx) * np.exp(-(tdx * tdx + (self.t_y - cy) ** 2) / (2 * R * R)) * s * 4 * (0.3 + self.t_w * 0.5)
        self._splat(E, self.t_x + dx, self.t_y + dy, self.t_b * shim)
        gb = self.g_b * (0.85 + 0.15 * np.sin(t * self.g_sp * 2 + self.g_ph))
        self._splat(E, self.g_x + gdx, self.g_y, gb)
        Fl = np.zeros_like(E)
        fr = self.g_fr
        self._splat(Fl, (self.g_x + gdx)[fr], self.g_y[fr], gb[fr])

        # ground mist
        off = int(t * 3.0) % gw
        mn = self.mist_noise[:, off:off + gw]
        mist_p = self.mist_band * np.clip(mn * 1.4 - 0.2, 0, 1) * 0.5
        mist_lit = r.random((gh, gw), dtype=np.float32) < mist_p
        E = np.maximum(E, np.where(mist_lit, 0.2 + 0.1 * mn, 0).astype(np.float32))

        # living particles
        self._emit_from_inputs(dt)
        self._drip_leaves(t)
        self._update_fireflies(t, dt, infl, X)
        self._update_sparks(t, dt)

        # the person
        P = np.zeros((gh, gw), np.float32)
        if mask is not None:
            if mask.mean() < 0.004:
                self.m_ema *= 0.85
            else:
                self.m_ema = 0.4 * self.m_ema + 0.6 * mask
        m = self.m_ema
        if m.max() > 0.02:
            p = np.clip((m - 0.3) / 0.3, 0, 1)
            p = p * p * (3 - 2 * p)
            p = p * self.sitfade
            rnd = r.random((gh, gw), dtype=np.float32)
            lit = (rnd < p * 0.93) & (r.random((gh, gw), dtype=np.float32) < self.rowfade)
            bright = 0.66 + 0.2 * r.random((gh, gw), dtype=np.float32)
            if luma is not None:
                bright += 0.35 * (luma - 0.45)
            blur = cv2.GaussianBlur(m, (0, 0), 2.0)
            bright += np.clip((m - blur) * 2.5, 0, 0.3)
            P = np.where(lit, np.clip(bright, 0.3, 0.82), 0).astype(np.float32)
            halo = (r.random((gh, gw), dtype=np.float32) < blur * 0.3) & ~lit
            P = np.maximum(P, np.where(halo, 0.38, 0).astype(np.float32))
            P = np.where(Fl > 0, P * 0.3, P)
            E *= 1 - 0.8 * cv2.GaussianBlur(p, (0, 0), 1.0)
            # embers lift off the silhouette edge
            edge = np.flatnonzero((lit & (blur < 0.72)).ravel())
            if edge.size:
                pick = r.choice(edge, size=min(4, edge.size), replace=False)
                ey, ex = np.divmod(pick, gw)
                nn = len(pick)
                self._emit(ex + r.normal(0, 0.4, nn), ey + r.normal(0, 0.4, nn),
                           r.normal(0, 3 * k, nn) + wind(ex.astype(np.float32), t) * 4 * k * wg,
                           -r.uniform(5, 14, nn) * k, nn, grav=-1.5, decay=(0.5, 1.0))

        # hands
        H = self._hand_layer(t)
        if H.max() > 0:
            E *= 1 - 0.6 * cv2.GaussianBlur((H > 0).astype(np.float32), (0, 0), 1.5)

        # sparks and drifting leaves
        if len(self.sx):
            v = (np.clip(self.sl, 0, 1) ** 1.3 * 0.95).astype(np.float32)
            ax = self.sacc
            if ax.any():
                self._glow_splat(X, self.sx[ax], self.sy[ax], v[ax])
                self._splat(X, self.sx[ax] - self.svx[ax] * dt * 1.6, self.sy[ax] - self.svy[ax] * dt * 1.6, v[ax] * 0.5)
            nx = ~ax
            if nx.any():
                self._splat(E, self.sx[nx], self.sy[nx], v[nx] * 0.8)

        # composite
        base = np.maximum(np.maximum(E * 1.35, P), H)
        out = np.maximum(base, X)
        use_accent = (X > base) & (X > 0.02)
        out *= 0.88 + 0.12 * r.random((gh, gw), dtype=np.float32)
        idx = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        col = self.cur_b[idx] + self.cur_d[idx] * self.vmap
        col = np.where(use_accent[..., None], self.cur_a[idx], col)
        return np.clip(col, 0, 255).astype(np.uint8)


class Renderer:
    """Turns the little colour grid into glowing, round, grainy dots."""

    def __init__(self, gw, gh, cell, seed=3):
        self.gw, self.gh, self.c = gw, gh, cell
        self.W, self.H = gw * cell, gh * cell
        c = cell
        yy, xx = np.mgrid[0:c, 0:c].astype(np.float32)
        d = np.hypot(xx - (c - 1) / 2, yy - (c - 1) / 2)
        m = np.clip((c * 0.42 + 0.7 - d) / 1.4, 0, 1)
        m = (0.05 + 0.95 * m) * 255
        tile = np.tile(m, (gh, gw)).astype(np.uint8)
        self.dots = np.dstack([tile] * 3)
        Y, X = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        rr = np.hypot((X / self.W - 0.5) * 1.1, (Y / self.H - 0.5) * 1.1)
        vig = np.clip(1.15 - rr * 1.1, 0.35, 1.0)
        self.vig = np.dstack([(vig * 255).astype(np.uint8)] * 3)
        rng = np.random.default_rng(seed)
        self.grain = []
        for _ in range(5):
            n = rng.integers(0, 8, (self.H, self.W), dtype=np.uint8)
            self.grain.append(np.dstack([n, (n * 0.45).astype(np.uint8), (n * 0.6).astype(np.uint8)]))
        self.k = 0

    def draw(self, col):
        W, H = self.W, self.H
        up = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
        out = cv2.multiply(up, self.dots, scale=1 / 255.0)
        b1 = cv2.resize(cv2.GaussianBlur(col, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_LINEAR)
        b2 = cv2.resize(cv2.GaussianBlur(col, (0, 0), 4.0), (W, H), interpolation=cv2.INTER_LINEAR)
        out = cv2.addWeighted(out, 1.0, b1, 0.35, 0)
        out = cv2.addWeighted(out, 1.0, b2, 0.4, 0)
        out = cv2.multiply(out, self.vig, scale=1 / 255.0)
        out = cv2.add(out, self.grain[self.k % len(self.grain)])
        self.k += 1
        return out
