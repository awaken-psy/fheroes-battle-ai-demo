"""Shared rendering utilities: Popup, button, unit drawing."""

import pygame

import config
from . import fonts


# ── Floating damage number ────────────────────────────────────

class Popup:
    """Floating text that rises and fades. `big` uses a larger font."""

    def __init__(self, x, y, text, color, life=1.0, speed=1.0, big=False):
        self.x, self.y = float(x), float(y)
        self.text = text
        self.color = color
        self.age = 0.0
        self.life = life
        self._speed = speed
        self.big = big

    def update(self, dt):
        self.age += dt
        self.y -= 22 * self._speed * dt
        return self.age < self.life

    def draw(self, surf):
        fade = max(0.0, 1.0 - self.age / self.life)
        c = tuple(int(ch * fade) for ch in self.color)
        font = fonts.BIG if self.big else fonts.POPUP
        txt = font.render(self.text, True, c)
        surf.blit(txt, txt.get_rect(center=(int(self.x), int(self.y))))


# ── Button ────────────────────────────────────────────────────

def draw_btn(surf, x, y, bw, bh, text, bg, fg):
    """Draw a rounded button with centred text."""
    r = pygame.Rect(x, y, bw, bh)
    pygame.draw.rect(surf, bg, r, border_radius=5)
    pygame.draw.rect(surf, (200, 200, 200), r, 1, border_radius=5)
    t = fonts.LABEL.render(text, True, fg)
    surf.blit(t, t.get_rect(center=r.center))


# ── Unit drawing ──────────────────────────────────────────────

def draw_unit(canvas, s, grid, u, cx, cy, current=False, selectable=False):
    """Render one unit (shape, symbol, hp bar, count) at pixel (cx, cy).

    ``cx, cy`` is the head cell's centre (or the animated position). A wide unit
    also fills its trailing tail cell, drawn underneath the head.
    ``selectable`` draws a pulsing green ring around the unit.
    """
    color = fonts.team_color(u.team)
    r = s(15)

    # Wide units span two cells: draw a body reaching into the tail cell first,
    # so the head shape (and symbol / bars) render on top of it.
    if u.is_wide and u.tail_cell is not None:
        tc, tr = u.tail_cell
        if 0 <= tc < grid.cols and 0 <= tr < grid.rows:
            hx, hy = grid.center(*u.pos)
            tx, ty = grid.center(*u.tail_cell)
            tcx, tcy = cx + (tx - hx), cy + (ty - hy)
            pygame.draw.line(canvas, color, (int(cx), int(cy)),
                             (int(tcx), int(tcy)), int(s(16)))
            pygame.draw.circle(canvas, color, (int(tcx), int(tcy)), int(s(11)))
            pygame.draw.circle(canvas, config.WHITE, (int(tcx), int(tcy)), int(s(11)), 1)

    if u.is_archer:
        pts = [(cx, cy - r), (cx - s(12), cy + s(10)), (cx + s(12), cy + s(10))]
        pygame.draw.polygon(canvas, color, pts)
        pygame.draw.polygon(canvas, config.WHITE, pts, 1)
    elif u.is_flying:
        pts = [(cx, cy - r), (cx + s(12), cy), (cx, cy + r), (cx - s(12), cy)]
        pygame.draw.polygon(canvas, color, pts)
        pygame.draw.polygon(canvas, config.WHITE, pts, 1)
    else:
        pygame.draw.circle(canvas, color, (int(cx), int(cy)), int(r))
        pygame.draw.circle(canvas, config.WHITE, (int(cx), int(cy)), int(r), 1)

    sym = fonts.LABEL.render(u.symbol, True, config.WHITE)
    canvas.blit(sym, sym.get_rect(center=(cx, cy - s(1))))

    hp_ratio = u._total_hp / u._max_total_hp
    bw = s(24); bx = cx - bw // 2; by = cy + s(18)
    pygame.draw.rect(canvas, (60, 20, 20), (bx, by, bw, s(6)))
    bar_c = config.GREEN if hp_ratio > 0.5 else (config.YELLOW if hp_ratio > 0.25 else config.RED)
    pygame.draw.rect(canvas, bar_c, (bx, by, max(1, int(bw * hp_ratio)), s(6)))

    info = f"{u.count}/{u._total_hp}"
    canvas.blit(fonts.DATA.render(info, True, config.WHITE), (cx + s(14), cy + s(8)))

    if current:
        pygame.draw.circle(canvas, config.YELLOW, (int(cx), int(cy)), int(s(18)), 2)

    if selectable:
        pygame.draw.circle(canvas, (60, 180, 60), (int(cx), int(cy)), int(s(20)), 2)
