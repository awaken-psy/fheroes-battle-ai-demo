"""Battle AI Demo — main entry point.

Controls:
  Setup:  click palette → click hex to place, right-click to remove
  Battle: Space=pause/step, 1/2/3=speed, R=reset, D=toggle AI debug
  Window: +/- zoom in/out, F11 toggle fullscreen, drag edge to resize
"""

import sys
import math
import pygame

import config
from hex_grid import HexGrid
from battle import Unit, BattleState, Action, MoveAction, AttackAction, SkipAction
from battle_ai import BattleAI

pygame.init()

# ── Font system ────────────────────────────────────────────
# JetBrains Mono for data/stats, Ubuntu Sans for text/labels
FONT_DATA = pygame.font.SysFont("jetbrainsmono", 12)   # stats, hints, small labels
FONT_BODY = pygame.font.SysFont("jetbrainsmono", 14)   # unit names, log, general text
FONT_POPUP = pygame.font.SysFont("jetbrainsmono", 22)  # damage numbers
FONT_LABEL = pygame.font.SysFont("ubuntusans", 15, bold=True)  # buttons, headers
FONT_TITLE = pygame.font.SysFont("ubuntusans", 18, bold=True)  # top bar, round info
FONT_BIG = pygame.font.SysFont("ubuntusans", 34, bold=True)    # game over text

# Backward-compatible aliases (to be replaced gradually)
FONT_SM = FONT_DATA
FONT_MD = FONT_BODY
FONT_LG = FONT_POPUP
FONT_XL = FONT_BIG

# Fixed virtual resolution — all rendering targets this surface
VW, VH = config.WINDOW_WIDTH, config.WINDOW_HEIGHT  # 1060 x 680
ASPECT = VW / VH  # ~1.559


# ── helpers ──────────────────────────────────────────────────

def team_color(team):
    return config.BLUE if team == 0 else config.RED

def team_light(team):
    return config.BLUE_LIGHT if team == 0 else config.RED_LIGHT

def team_name(team):
    return "Blue" if team == 0 else "Red"


# ── Game ─────────────────────────────────────────────────────

SETUP, BATTLE, GAME_OVER = 0, 1, 2

# Animation phases
PH_IDLE, PH_MOVE, PH_STRIKE, PH_RETAL, PH_AFTER = range(5)


class Popup:
    """Floating damage number that rises and fades."""
    def __init__(self, x, y, text, color, life=1.0, speed=1.0):
        self.x, self.y = float(x), float(y)
        self.text = text
        self.color = color
        self.age = 0.0
        self.life = life
        self._speed = speed

    def update(self, dt):
        self.age += dt
        self.y -= 22 * self._speed * dt
        return self.age < self.life

    def draw(self, surf):
        fade = max(0.0, 1.0 - self.age / self.life)
        c = tuple(int(ch * fade) for ch in self.color)
        txt = FONT_POPUP.render(self.text, True, c)
        surf.blit(txt, txt.get_rect(center=(int(self.x), int(self.y))))


class Game:
    def __init__(self):
        self.fullscreen = False
        # start at 3.5x canvas size (27" display)
        self.win_w, self.win_h = int(VW * 3.5), int(VH * 3.5)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("HoMM2 Battle AI Demo")
        self.clock = pygame.time.Clock()

        # render scale: virtual coords → native window pixels
        self._rs = self.win_w / VW
        # native-resolution canvas — no blurry scaling
        self.canvas = pygame.Surface((self.win_w, self.win_h))
        self._init_fonts()
        self._init_grid()

        self.ai = BattleAI()

        # setup state
        self.units: list[Unit] = []
        self.sel_type: str | None = None
        self.sel_team = 0
        self.hover: tuple | None = None

        # battle state
        self.battle: BattleState | None = None
        self.b_action: Action | None = None
        self.b_desc = ""
        self.b_log: list[str] = []
        self.b_path: set | None = None
        self.b_target: Unit | None = None
        self.speed = 2
        self.paused = False
        self.debug = True
        self.state = SETUP

        # animation state
        self._ph = PH_IDLE
        self._ph_t = 0.0
        self._anim_unit = None
        self._anim_px = (0.0, 0.0)
        self._move_px = []
        self._move_idx = 0
        self._move_frac = 0.0
        self._popups: list[Popup] = []
        self._projectile = None
        self._flash = None
        self._exec_result = None

    # ── scale helpers ───────────────────────────────────────

    def _s(self, v):
        """Scale virtual coordinate/size to native pixels."""
        return v * self._rs

    def _init_fonts(self):
        """Create fonts scaled to native resolution."""
        rs = self._rs
        global FONT_DATA, FONT_BODY, FONT_LABEL, FONT_TITLE, FONT_POPUP, FONT_BIG
        global FONT_SM, FONT_MD, FONT_LG, FONT_XL
        FONT_DATA = pygame.font.SysFont("jetbrainsmono", max(10, int(12 * rs)))
        FONT_BODY = pygame.font.SysFont("jetbrainsmono", max(12, int(14 * rs)))
        FONT_LABEL = pygame.font.SysFont("ubuntusans", max(12, int(15 * rs)), bold=True)
        FONT_TITLE = pygame.font.SysFont("ubuntusans", max(14, int(18 * rs)), bold=True)
        FONT_POPUP = pygame.font.SysFont("jetbrainsmono", max(16, int(22 * rs)))
        FONT_BIG = pygame.font.SysFont("ubuntusans", max(24, int(34 * rs)), bold=True)
        FONT_SM = FONT_DATA
        FONT_MD = FONT_BODY
        FONT_LG = FONT_POPUP
        FONT_XL = FONT_BIG

    def _init_grid(self):
        """Create hex grid at native resolution."""
        self.grid = HexGrid(self._rs)

    def _rebuild_canvas(self):
        """Rebuild canvas, fonts, grid after window size change."""
        self._rs = self.win_w / VW
        self.canvas = pygame.Surface((self.win_w, self.win_h))
        self._init_fonts()
        self._init_grid()

    # ── window sizing ────────────────────────────────────────

    def _apply_window_size(self):
        """Recreate the display at (win_w, win_h), locked aspect ratio."""
        self.win_h = max(360, self.win_h)
        self.win_w = max(560, int(self.win_h * ASPECT))
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            self.win_w, self.win_h = info.current_w, info.current_h
        else:
            self.screen = pygame.display.set_mode(
                (self.win_w, self.win_h), pygame.RESIZABLE
            )
        self._rebuild_canvas()

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._apply_window_size()

    def _scale_canvas_to_screen(self):
        """Blit native-resolution canvas directly to screen."""
        self.screen.blit(self.canvas, (0, 0))

    def _screen_to_canvas(self, pos: tuple) -> tuple:
        """Convert screen pixel coords to native canvas coords (= identity)."""
        return pos

    # ── main loop ────────────────────────────────────────────

    def run(self):
        frame = 0
        while True:
            dt = self.clock.tick(config.FPS) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.win_w, self.win_h = ev.w, ev.h
                    self._apply_window_size()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F12:
                    pygame.image.save(self.canvas, '/tmp/demo-screenshot.png')
                    print('Screenshot saved to /tmp/demo-screenshot.png')
                self._handle(ev)
            self._update(dt)
            self._draw()
            frame += 1

    # ── event handling ───────────────────────────────────────

    def _handle(self, ev):
        # translate mouse coords to canvas space
        if hasattr(ev, 'pos'):
            ev_dict = dict(ev.__dict__)
            ev_dict['pos'] = self._screen_to_canvas(ev.pos)
            # rebuild a compatible event-like object (type is the first ctor arg)
            canvas_ev = pygame.event.Event(ev.type, **{k: v for k, v in ev_dict.items()
                                                         if k != 'type'})
        else:
            canvas_ev = ev

        if self.state == SETUP:
            self._handle_setup(canvas_ev)
        elif self.state == BATTLE:
            self._handle_battle(canvas_ev)
        elif self.state == GAME_OVER:
            if canvas_ev.type == pygame.MOUSEBUTTONDOWN and canvas_ev.button == 1:
                if self._playagain_rect().collidepoint(canvas_ev.pos):
                    self._reset()

        # global keys (applied to raw event, no coord transform needed)
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.win_h += 60
                self._apply_window_size()
            elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.win_h -= 60
                self._apply_window_size()
            elif ev.key == pygame.K_F11:
                self._toggle_fullscreen()

    def _handle_setup(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.grid.pixel_to_hex(*ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            if ev.button == 1:
                for i, name in enumerate(config.UNIT_TYPES):
                    if self._palette_rect(i).collidepoint(mx, my):
                        self.sel_type = name; return
                if pygame.Rect(int(self._s(14)), int(self._s(12)),
                               int(self._s(100)), int(self._s(32))).collidepoint(mx, my):
                    self.sel_team = 1 - self.sel_team; return
                for i, pname in enumerate(config.PRESETS):
                    if self._preset_rect(i).collidepoint(mx, my):
                        self._load_preset(pname); return
                if self._start_btn_rect().collidepoint(mx, my):
                    if self._can_start(): self._start_battle()
                    return
                hex_pos = self.grid.pixel_to_hex(mx, my)
                if hex_pos and self.sel_type:
                    col, row = hex_pos
                    if self.grid.half_of(col) == self.sel_team:
                        self.units = [u for u in self.units if u.pos != hex_pos]
                        self.units.append(Unit.from_type(self.sel_type, self.sel_team, col, row))
            elif ev.button == 3:
                hex_pos = self.grid.pixel_to_hex(mx, my)
                if hex_pos:
                    self.units = [u for u in self.units if u.pos != hex_pos]

    def _handle_battle(self, ev):
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif ev.key == pygame.K_1:
                self.speed = 0
            elif ev.key == pygame.K_2:
                self.speed = 1
            elif ev.key == pygame.K_3:
                self.speed = 2
            elif ev.key == pygame.K_r:
                self._reset()
            elif ev.key == pygame.K_d:
                self.debug = not self.debug

    # ── update ───────────────────────────────────────────────

    def _update(self, dt):
        if self.state != BATTLE:
            return
        # always update popups and flash (purely visual)
        self._popups = [p for p in self._popups if p.update(dt)]
        if self._flash:
            pos, timer = self._flash
            timer -= dt
            self._flash = (pos, timer) if timer > 0 else None
        if self.paused:
            return

        spd = [1.5, 1.0, 0.5][self.speed]

        if self._ph == PH_IDLE:
            self._next_unit()
        elif self._ph == PH_MOVE:
            self._anim_move(dt, spd)
        elif self._ph == PH_STRIKE:
            self._anim_strike(dt, spd)
        elif self._ph == PH_RETAL:
            self._ph_t += dt
            if self._ph_t >= 0.35 * spd:
                self._ph = PH_AFTER; self._ph_t = 0
        elif self._ph == PH_AFTER:
            self._ph_t += dt
            if self._ph_t >= 0.25 * spd:
                self._anim_unit = None; self._exec_result = None
                self.b_path = None; self.b_target = None
                if self.battle.is_over():
                    self.state = GAME_OVER; return
                self._ph = PH_IDLE

    def _next_unit(self):
        if not self.battle.alive():
            return
        order = self.battle.turn_order()
        if not hasattr(self, '_round_order') or self._round_order != order or self._round_num != self.battle.round_num:
            self._round_order = order
            self._order_idx = 0
            self.battle.start_round()
            self._round_num = self.battle.round_num
        while self._order_idx < len(self._round_order):
            unit = self._round_order[self._order_idx]
            self._order_idx += 1
            if unit.is_alive:
                action, desc = self.ai.decide(self.battle, unit)
                self.b_action = action; self.b_desc = desc
                self._start_anim(action)
                return
        self._round_order = None; self._next_unit()

    # ── animation engine ─────────────────────────────────────

    def _start_anim(self, action):
        """Initialize animation for the decided action."""
        self._exec_result = None
        self._projectile = None

        if isinstance(action, SkipAction):
            self._anim_unit = action.unit
            self._anim_px = self.grid.center(*action.unit.pos)
            self._ph = PH_AFTER; self._ph_t = 0
            return

        if isinstance(action, MoveAction):
            self._anim_unit = action.unit
            self._move_px = [self.grid.center(*p) for p in action.path]
            self._move_idx = 0; self._move_frac = 0.0
            self._anim_px = self._move_px[0]
            self.b_path = set(action.path)
            self._ph = PH_MOVE; self._ph_t = 0
            return

        if isinstance(action, AttackAction):
            self._anim_unit = action.attacker
            self.b_target = action.target
            if action.ranged:
                # no movement, go straight to strike
                self._anim_px = self.grid.center(*action.attacker.pos)
                self._ph = PH_STRIKE; self._ph_t = 0
            else:
                # melee: move to attack position first
                if action.from_pos and action.from_pos != action.attacker.pos:
                    occ = self.battle.occupied(exclude=action.attacker)
                    path = self.grid.find_path(
                        action.attacker.pos, action.from_pos, occ,
                        action.attacker.is_flying, action.attacker.speed)
                    self._move_px = ([self.grid.center(*p) for p in path]
                                     if path
                                     else [self.grid.center(*action.attacker.pos),
                                           self.grid.center(*action.from_pos)])
                    self.b_path = {action.from_pos}
                else:
                    self._move_px = [self.grid.center(*action.attacker.pos)]
                self._move_idx = 0; self._move_frac = 0.0
                self._anim_px = self._move_px[0]
                self._ph = PH_MOVE; self._ph_t = 0

    def _anim_move(self, dt, spd):
        """Animate unit sliding along path."""
        px_per_sec = 280.0 / spd

        if self._move_idx >= len(self._move_px) - 1:
            self._anim_px = self._move_px[-1]
            if isinstance(self.b_action, AttackAction):
                self._ph = PH_STRIKE; self._ph_t = 0
            else:
                # MoveAction: execute now
                result = self.battle.execute(self.b_action)
                self.b_log.append(result['desc'])
                if len(self.b_log) > 5: self.b_log.pop(0)
                self._ph = PH_AFTER; self._ph_t = 0
            return

        a = self._move_px[self._move_idx]
        b = self._move_px[self._move_idx + 1]
        seg_len = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1.0)
        self._move_frac += px_per_sec * dt / seg_len

        while self._move_frac >= 1.0 and self._move_idx < len(self._move_px) - 1:
            self._move_frac -= 1.0
            self._move_idx += 1

        if self._move_idx >= len(self._move_px) - 1:
            self._anim_px = self._move_px[-1]
        else:
            a = self._move_px[self._move_idx]
            b = self._move_px[self._move_idx + 1]
            t = min(self._move_frac, 1.0)
            self._anim_px = (a[0] + (b[0] - a[0]) * t,
                             a[1] + (b[1] - a[1]) * t)

    def _anim_strike(self, dt, spd):
        """Animate attack: melee lunge or ranged projectile."""
        action = self.b_action
        self._ph_t += dt

        if action.ranged:
            lunge_dur = 0.25 * spd
            src = self.grid.center(*action.attacker.pos)
            dst = self.grid.center(*action.target.pos)
            prog = min(1.0, self._ph_t / lunge_dur)
            self._projectile = (src, dst, prog)

            if self._ph_t >= lunge_dur:
                self._projectile = None
                self._exec_result = self.battle.execute(self.b_action)
                self.b_log.append(self._exec_result['desc'])
                if len(self.b_log) > 5: self.b_log.pop(0)
                # damage popup on target
                self._popups.append(
                    Popup(dst[0], dst[1] - self._s(12),
                          f"-{self._exec_result['dmg']}", config.RED, speed=self._rs))
                if not self._exec_result['target_alive']:
                    self._popups.append(
                        Popup(dst[0], dst[1] - self._s(32), "DEAD", config.YELLOW, speed=self._rs))
                self._flash = (action.target.pos, 0.15)
                self._goto_retal_or_after()
        else:
            # melee lunge
            lunge_dur = 0.15 * spd
            ret_dur = 0.12 * spd
            atk_px = self._move_px[-1] if self._move_px else self.grid.center(*action.attacker.pos)
            tgt_px = self.grid.center(*action.target.pos)
            dx = tgt_px[0] - atk_px[0]
            dy = tgt_px[1] - atk_px[1]

            if self._ph_t < lunge_dur:
                # lunge toward target (35% of distance, smoothstep)
                t = self._ph_t / lunge_dur
                t = t * t * (3 - 2 * t)
                self._anim_px = (atk_px[0] + dx * 0.35 * t,
                                 atk_px[1] + dy * 0.35 * t)
            elif self._ph_t < lunge_dur + ret_dur:
                # execute on first frame of return phase
                if self._exec_result is None:
                    self._exec_result = self.battle.execute(self.b_action)
                    self.b_log.append(self._exec_result['desc'])
                    if len(self.b_log) > 5: self.b_log.pop(0)
                    self._popups.append(
                        Popup(tgt_px[0], tgt_px[1] - self._s(12),
                              f"-{self._exec_result['dmg']}", config.RED, speed=self._rs))
                    if not self._exec_result['target_alive']:
                        self._popups.append(
                            Popup(tgt_px[0], tgt_px[1] - self._s(32), "DEAD", config.YELLOW, speed=self._rs))
                    self._flash = (action.target.pos, 0.15)
                # return from lunge
                t = min(1.0, (self._ph_t - lunge_dur) / ret_dur)
                self._anim_px = (atk_px[0] + dx * 0.35 * (1 - t),
                                 atk_px[1] + dy * 0.35 * (1 - t))
            else:
                self._anim_px = atk_px
                self._goto_retal_or_after()

    def _goto_retal_or_after(self):
        """Transition to retaliation pause or after phase."""
        r = self._exec_result
        if r and r['ret_dmg'] > 0:
            atk = self.b_action.attacker
            ax, ay = self.grid.center(*atk.pos)
            self._popups.append(
                Popup(ax, ay - self._s(12), f"-{r['ret_dmg']}", config.ORANGE, speed=self._rs))
            if not r['attacker_alive']:
                self._popups.append(
                    Popup(ax, ay - self._s(32), "DEAD", config.YELLOW, speed=self._rs))
            self._flash = (atk.pos, 0.15)
            self._ph = PH_RETAL; self._ph_t = 0
        else:
            self._ph = PH_AFTER; self._ph_t = 0

    # ── drawing (all to self.canvas at VW×VH) ────────────────

    def _draw(self):
        self.canvas.fill(config.BG)
        if self.state == SETUP:
            self._draw_setup()
        elif self.state == BATTLE:
            self._draw_battle()
        elif self.state == GAME_OVER:
            self._draw_gameover()
        self._scale_canvas_to_screen()
        pygame.display.flip()

    def _draw_btn(self, x, y, bw, bh, text, bg, fg, surf=None):
        s = surf or self.canvas
        r = pygame.Rect(x, y, bw, bh)
        pygame.draw.rect(s, bg, r, border_radius=5)
        pygame.draw.rect(s, (200, 200, 200), r, 1, border_radius=5)
        t = FONT_LABEL.render(text, True, fg)
        s.blit(t, t.get_rect(center=r.center))

    # ── layout constants (virtual canvas coords) ─────────────

    @property
    def bottom_y(self):
        return self._s(VH - 100)

    def _palette_rect(self, i):
        return pygame.Rect(self._s(12), self._s(74 + i * 56), self._s(200), self._s(50))

    def _start_btn_rect(self):
        cx = self._s(VW) // 2
        return pygame.Rect(cx - self._s(100), self._s(VH - 55), self._s(200), self._s(44))

    def _playagain_rect(self):
        cx = self._s(VW) // 2
        return pygame.Rect(cx - self._s(100), self._s(VH) // 2 + self._s(30), self._s(200), self._s(48))

    def _preset_rect(self, i):
        total = len(config.PRESETS)
        return pygame.Rect(self._s(12), self._s(VH - 100 - 10 - (total - i) * 32), self._s(200), self._s(28))

    # ── setup drawing ────────────────────────────────────────

    def _draw_setup(self):
        s = self._s
        # Reposition grid to center in available space after left panel
        panel_right = s(228)  # panel ends at ~220 + 8px gap
        grid_w = self.grid.cols * self.grid.hex_w
        avail = self.win_w - panel_right
        self.grid.reposition(panel_right + (avail - grid_w) / 2, s(config.GRID_OFFSET_Y))

        # ── left panel background ──
        panel = pygame.Rect(int(s(4)), int(s(4)), int(s(216)), int(self._s(VH) - s(8)))
        pygame.draw.rect(self.canvas, config.PANEL_BG, panel, border_radius=int(s(6)))
        pygame.draw.rect(self.canvas, (55, 65, 90), panel, 1, border_radius=int(s(6)))

        # team selector
        self._draw_btn(s(14), s(12), s(100), s(32),
                       f"Team: {team_name(self.sel_team)}",
                       team_color(self.sel_team), config.WHITE)
        txt = FONT_BODY.render("Click palette -> hex. Right-click remove.", True, (170, 180, 200))
        self.canvas.blit(txt, (s(230), s(16)))

        # section header
        self.canvas.blit(FONT_TITLE.render("UNITS", True, config.WHITE), (s(18), s(52)))

        for i, name in enumerate(config.UNIT_TYPES):
            r = self._palette_rect(i)
            sel = self.sel_type == name
            bg = (50, 55, 78) if sel else (38, 45, 65)
            pygame.draw.rect(self.canvas, bg, r, border_radius=int(s(4)))
            if sel:
                pygame.draw.rect(self.canvas, config.YELLOW, r, 2, border_radius=int(s(4)))
            else:
                pygame.draw.rect(self.canvas, (60, 68, 92), r, 1, border_radius=int(s(4)))
            ut = config.UNIT_TYPES[name]
            tc = team_light(self.sel_team)
            self.canvas.blit(FONT_LABEL.render(ut["symbol"], True, tc), (r.x + s(8), r.y + s(5)))
            self.canvas.blit(FONT_BODY.render(name, True, config.WHITE), (r.x + s(28), r.y + s(4)))
            self.canvas.blit(FONT_DATA.render(
                f"A{ut['attack']} D{ut['defense']} H{ut['hp']} S{ut['speed']} x{ut['count']}",
                True, (170, 180, 200)), (r.x + s(28), r.y + s(24)))

        # separator
        sep_y = self._preset_rect(0).y - s(8)
        pygame.draw.line(self.canvas, (55, 65, 90), (s(14), sep_y), (s(210), sep_y), 1)
        self.canvas.blit(FONT_LABEL.render("PRESETS", True, config.WHITE), (s(18), sep_y - s(20)))

        # hex grid
        highlights = {}
        if self.hover and self.sel_type:
            if self.grid.half_of(self.hover[0]) == self.sel_team:
                highlights[self.hover] = team_color(self.sel_team)
        self.grid.draw_grid(self.canvas, highlights)
        self._draw_units(self.units)

        # start button
        can = self._can_start()
        sr = self._start_btn_rect()
        self._draw_btn(sr.x, sr.y, sr.w, sr.h, "Start Battle",
                       config.GREEN if can else config.GRAY, config.BLACK)

        for i, pname in enumerate(config.PRESETS):
            r = self._preset_rect(i)
            pygame.draw.rect(self.canvas, config.DARK, r, border_radius=int(s(3)))
            pygame.draw.rect(self.canvas, (60, 68, 92), r, 1, border_radius=int(s(3)))
            self.canvas.blit(FONT_BODY.render(f"Preset: {pname}", True, config.WHITE), (r.x + s(8), r.y + s(5)))

        for team in (0, 1):
            n = sum(1 for u in self.units if u.team == team)
            txt = FONT_BODY.render(f"{team_name(team)}: {n} units", True, team_light(team))
            if team == 0:
                self.canvas.blit(txt, (int(self.grid.ox) + int(s(10)), int(s(56))))
            else:
                self.canvas.blit(txt, (int(self.grid.ox) + int(grid_w) - txt.get_width() - int(s(10)), int(s(56))))

    def _draw_units(self, units, current=None):
        s = self._s
        for u in units:
            if not u.is_alive:
                continue
            if u is self._anim_unit and self._anim_px:
                cx, cy = self._anim_px
            else:
                cx, cy = self.grid.center(*u.pos)
            color = team_color(u.team)
            r = s(15)  # unit shape radius
            if u.is_archer:
                pts = [(cx, cy - r), (cx - s(12), cy + s(10)), (cx + s(12), cy + s(10))]
                pygame.draw.polygon(self.canvas, color, pts)
                pygame.draw.polygon(self.canvas, config.WHITE, pts, 1)
            elif u.is_flying:
                pts = [(cx, cy - r), (cx + s(12), cy), (cx, cy + r), (cx - s(12), cy)]
                pygame.draw.polygon(self.canvas, color, pts)
                pygame.draw.polygon(self.canvas, config.WHITE, pts, 1)
            else:
                pygame.draw.circle(self.canvas, color, (int(cx), int(cy)), int(r))
                pygame.draw.circle(self.canvas, config.WHITE, (int(cx), int(cy)), int(r), 1)
            sym = FONT_LABEL.render(u.symbol, True, config.WHITE)
            self.canvas.blit(sym, sym.get_rect(center=(cx, cy - s(1))))
            hp_ratio = u._total_hp / (u.max_hp * max(u.count, 1))
            bw = s(24); bx = cx - bw // 2; by = cy + s(18)
            pygame.draw.rect(self.canvas, (60, 20, 20), (bx, by, bw, s(6)))
            c = config.GREEN if hp_ratio > 0.5 else (config.YELLOW if hp_ratio > 0.25 else config.RED)
            pygame.draw.rect(self.canvas, c, (bx, by, max(1, int(bw * hp_ratio)), s(6)))
            self.canvas.blit(FONT_DATA.render(str(u.count), True, config.WHITE), (cx + s(14), cy + s(8)))
            if u is current:
                pygame.draw.circle(self.canvas, config.YELLOW, (int(cx), int(cy)), int(s(18)), 2)

    # ── battle drawing ───────────────────────────────────────

    def _draw_battle(self):
        s = self._s
        # Reposition grid to center horizontally on screen
        grid_w = self.grid.cols * self.grid.hex_w
        self.grid.reposition((self.win_w - grid_w) / 2, s(config.GRID_OFFSET_Y))

        # ── top bar background ──
        top_bar = pygame.Rect(0, 0, s(VW), s(42))
        pygame.draw.rect(self.canvas, config.PANEL_BG, top_bar)
        pygame.draw.line(self.canvas, (55, 65, 90), (0, s(42)), (s(VW), s(42)), 1)

        if self.battle:
            cx = int(self.win_w // 2)
            bar_cy = int(s(21))  # vertical center of top bar
            # Round text centered at screen center
            round_surf = FONT_BIG.render(f"Round {self.battle.round_num}", True, config.WHITE)
            round_rect = round_surf.get_rect(center=(cx, bar_cy))
            self.canvas.blit(round_surf, round_rect)
            # Subtle vertical dividers flanking the round text
            div_x_l = round_rect.left - int(s(12))
            div_x_r = round_rect.right + int(s(12))
            pygame.draw.line(self.canvas, (55, 65, 90),
                             (div_x_l, int(s(8))), (div_x_l, int(s(34))), 1)
            pygame.draw.line(self.canvas, (55, 65, 90),
                             (div_x_r, int(s(8))), (div_x_r, int(s(34))), 1)
            # Team info symmetric around center, with colored dot indicator
            # Use divider positions as boundaries so text never overlaps Round
            dot_r = int(s(5))
            dot_gap = int(s(4))
            div_pad = int(s(10))  # padding from divider to text
            for team in (0, 1):
                units = self.battle.alive(team)
                total = sum(u.strength for u in units)
                n = len(units)
                label = f"{team_name(team)}: {n} units  STR {total:.0f}"
                txt = FONT_TITLE.render(label, True, team_light(team))
                if team == 0:
                    rect = txt.get_rect(right=div_x_l - div_pad, centery=bar_cy)
                    pygame.draw.circle(self.canvas, team_color(team),
                                       (rect.left - dot_r - dot_gap, bar_cy), dot_r)
                else:
                    rect = txt.get_rect(left=div_x_r + div_pad, centery=bar_cy)
                    pygame.draw.circle(self.canvas, team_color(team),
                                       (rect.left - dot_r - dot_gap, bar_cy), dot_r)
                self.canvas.blit(txt, rect)

        highlights = {}
        if self.b_path:
            for p in self.b_path:
                highlights.setdefault(p, config.PATH_COLOR)
        if self.b_action and isinstance(self.b_action, MoveAction):
            for p in self.b_action.path:
                highlights[p] = tuple(min(highlights.get(p, config.BG)[i] + 40, 255) for i in range(3))
        if self._flash and self._flash[1] > 0:
            intensity = int(min(255, 160 * self._flash[1] / 0.15))
            highlights[self._flash[0]] = (intensity, 40, 40)
        self.grid.draw_grid(self.canvas, highlights)

        if self.b_target and self.b_action and isinstance(self.b_action, AttackAction):
            self.grid.draw_dashed_line(self.canvas, self.b_action.attacker.pos,
                                       self.b_target.pos, config.TARGET_COLOR, 2)
            self.grid.draw_overlay(self.canvas, self.b_target.pos, config.TARGET_COLOR, 3)

        # projectile (ranged attack in flight)
        if self._projectile:
            src, dst, prog = self._projectile
            ex = src[0] + (dst[0] - src[0]) * prog
            ey = src[1] + (dst[1] - src[1]) * prog
            pygame.draw.line(self.canvas, config.YELLOW,
                             (int(src[0]), int(src[1])), (int(ex), int(ey)), 2)
            pygame.draw.circle(self.canvas, config.YELLOW, (int(ex), int(ey)), 4)

        current_unit = None
        if self.b_action:
            if isinstance(self.b_action, MoveAction): current_unit = self.b_action.unit
            elif isinstance(self.b_action, AttackAction): current_unit = self.b_action.attacker
            elif isinstance(self.b_action, SkipAction): current_unit = self.b_action.unit
        self._draw_units(self.units, current=current_unit)

        # floating damage numbers
        for p in self._popups:
            p.draw(self.canvas)

        if self.debug and self.b_desc:
            bar = pygame.Rect(0, self.bottom_y, s(VW), s(VH) - self.bottom_y)
            pygame.draw.rect(self.canvas, config.PANEL_BG, bar)
            pygame.draw.line(self.canvas, (55, 65, 90), bar.topleft, bar.topright, 2)
            for i, line in enumerate(self.b_desc.split(" -> ")):
                last = (i == len(self.b_desc.split(" -> ")) - 1)
                self.canvas.blit(FONT_BODY.render(line, True, config.CYAN if last else config.WHITE),
                                 (s(14), self.bottom_y + s(8) + i * s(20)))
            for i, log in enumerate(self.b_log):
                self.canvas.blit(FONT_DATA.render(log, True, config.GRAY),
                                 (s(VW) // 2, self.bottom_y + s(8) + i * s(16)))

        spd_names = ["Slow", "Normal", "Fast"]
        hints = (f"[Space] {'>> Play' if self.paused else '|| Pause'}   "
                 f"[1/2/3] Speed: {spd_names[self.speed]}   "
                 f"[R] Reset   [D] Debug: {'ON' if self.debug else 'OFF'}   "
                 f"[+/-] Size  [F11] Fullscreen")
        # hint bar at very bottom
        hint_y = s(VH) - s(22)
        pygame.draw.rect(self.canvas, config.PANEL_BG, (0, hint_y - s(4), s(VW), s(26)))
        pygame.draw.line(self.canvas, (55, 65, 90), (0, hint_y - s(4)), (s(VW), hint_y - s(4)), 1)
        self.canvas.blit(FONT_DATA.render(hints, True, config.GRAY), (s(14), hint_y))

    # ── game over ────────────────────────────────────────────

    def _draw_gameover(self):
        self._draw_battle()
        s = self._s
        overlay = pygame.Surface((s(VW), s(VH)), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.canvas.blit(overlay, (0, 0))
        if self.battle:
            w = self.battle.winner()
            txt = FONT_BIG.render(f"{team_name(w)} Wins!", True, team_color(w))
            self.canvas.blit(txt, txt.get_rect(center=(s(VW) // 2, s(VH) // 2 - s(20))))
        r = self._playagain_rect()
        self._draw_btn(r.x, r.y, r.w, r.h, "Play Again", config.GREEN, config.BLACK)

    # ── game logic ───────────────────────────────────────────

    def _can_start(self):
        return any(u.team == 0 for u in self.units) and any(u.team == 1 for u in self.units)

    def _start_battle(self):
        self.battle = BattleState(self.grid, self.units)
        self.state = BATTLE
        self._ph = PH_IDLE; self.b_log = []; self._popups = []
        self._round_order = None; self._order_idx = 0; self._round_num = 0

    def _load_preset(self, name):
        preset = config.PRESETS[name]
        self.units = []
        for team, placements in preset.items():
            for type_name, col, row in placements:
                self.units.append(Unit.from_type(type_name, team, col, row))

    def _reset(self):
        self.state = SETUP; self.battle = None
        self.b_action = None; self.b_desc = ""
        self.b_path = None; self.b_target = None; self.b_log = []
        self.units = []; self._round_order = None
        self._ph = PH_IDLE; self._anim_unit = None
        self._popups = []; self._projectile = None; self._flash = None


if __name__ == "__main__":
    Game().run()
