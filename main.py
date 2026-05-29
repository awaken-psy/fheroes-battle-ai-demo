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

FONT_SM = pygame.font.SysFont("consolas", 13)
FONT_MD = pygame.font.SysFont("consolas", 16)
FONT_LG = pygame.font.SysFont("consolas", 22)
FONT_XL = pygame.font.SysFont("consolas", 36)

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
B_IDLE, B_SHOW, B_EXEC, B_RESULT = 0, 1, 2, 3


class Game:
    def __init__(self):
        self.fullscreen = False
        # start at 2.5x canvas size (27" display)
        self.win_w, self.win_h = int(VW * 3.5), int(VH * 3.5)
        self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("HoMM2 Battle AI Demo")
        self.clock = pygame.time.Clock()

        # fixed virtual surface — everything draws here
        self.canvas = pygame.Surface((VW, VH))

        self.grid = HexGrid()
        self.ai = BattleAI()

        # setup state
        self.units: list[Unit] = []
        self.sel_type: str | None = None
        self.sel_team = 0
        self.hover: tuple | None = None

        # battle state
        self.battle: BattleState | None = None
        self.b_sub = B_IDLE
        self.b_timer = 0.0
        self.b_action: Action | None = None
        self.b_desc = ""
        self.b_log: list[str] = []
        self.b_path: set | None = None
        self.b_target: Unit | None = None
        self.speed = 2
        self.paused = False
        self.debug = True
        self.state = SETUP

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

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._apply_window_size()

    def _scale_canvas_to_screen(self):
        """Scale the fixed canvas to fill the window, letterboxed."""
        sw = self.win_w / VW
        sh = self.win_h / VH
        scale = min(sw, sh)
        dst_w = int(VW * scale)
        dst_h = int(VH * scale)
        dst_x = (self.win_w - dst_w) // 2
        dst_y = (self.win_h - dst_h) // 2
        self.screen.fill(config.BLACK)
        scaled = pygame.transform.scale(self.canvas, (dst_w, dst_h))
        self.screen.blit(scaled, (dst_x, dst_y))

    def _screen_to_canvas(self, pos: tuple) -> tuple:
        """Convert screen pixel coords to canvas coords."""
        sw = self.win_w / VW
        sh = self.win_h / VH
        scale = min(sw, sh)
        dst_w = int(VW * scale)
        dst_h = int(VH * scale)
        dst_x = (self.win_w - dst_w) // 2
        dst_y = (self.win_h - dst_h) // 2
        return ((pos[0] - dst_x) / scale, (pos[1] - dst_y) / scale)

    # ── main loop ────────────────────────────────────────────

    def run(self):
        while True:
            dt = self.clock.tick(config.FPS) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.win_w, self.win_h = ev.w, ev.h
                    self._apply_window_size()
                self._handle(ev)
            self._update(dt)
            self._draw()

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
                if pygame.Rect(10, 8, 90, 28).collidepoint(mx, my):
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
        if self.paused and self.b_sub != B_IDLE:
            return
        speeds = [1.5, 0.7, 0.25]
        interval = speeds[self.speed]
        self.b_timer += dt

        if self.b_sub == B_IDLE:
            self._next_unit()
        elif self.b_sub == B_SHOW and self.b_timer >= interval:
            self.b_timer = 0
            desc = self.battle.execute(self.b_action)
            self.b_log.append(desc)
            if len(self.b_log) > 5:
                self.b_log.pop(0)
            self.b_sub = B_EXEC
        elif self.b_sub == B_EXEC and self.b_timer >= interval * 0.6:
            self.b_timer = 0
            if self.battle.is_over():
                self.state = GAME_OVER; return
            self.b_path = None; self.b_target = None
            self.b_sub = B_IDLE
        elif self.b_sub == B_RESULT and self.b_timer >= interval * 0.3:
            self.b_timer = 0; self.b_sub = B_IDLE

    def _next_unit(self):
        if not self.battle.alive():
            return
        order = self.battle.turn_order()
        if not hasattr(self, '_round_order') or self._round_order != order or self._round_num != self.battle.round_num:
            self._round_order = order
            self._round_num = self.battle.round_num
            self._order_idx = 0
            self.battle.start_round()
        while self._order_idx < len(self._round_order):
            unit = self._round_order[self._order_idx]
            self._order_idx += 1
            if unit.is_alive:
                action, desc = self.ai.decide(self.battle, unit)
                self.b_action = action; self.b_desc = desc
                self.b_path = None; self.b_target = None
                if isinstance(action, MoveAction):
                    self.b_path = set(action.path)
                elif isinstance(action, AttackAction):
                    if action.from_pos:
                        self.b_path = {action.from_pos}
                    self.b_target = action.target
                self.b_timer = 0; self.b_sub = B_SHOW; return
        self._round_order = None; self._next_unit()

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
        t = FONT_MD.render(text, True, fg)
        s.blit(t, t.get_rect(center=r.center))

    # ── layout constants (virtual canvas coords) ─────────────

    @property
    def bottom_y(self):
        return VH - 90  # bottom debug bar height

    def _palette_rect(self, i):
        return pygame.Rect(10, 70 + i * 50, 190, 44)

    def _start_btn_rect(self):
        return pygame.Rect(VW // 2 - 90, VH - 50, 180, 40)

    def _playagain_rect(self):
        return pygame.Rect(VW // 2 - 90, VH // 2 + 30, 180, 44)

    def _preset_rect(self, i):
        total = len(config.PRESETS)
        return pygame.Rect(10, VH - 90 - 10 - (total - i) * 28, 190, 24)

    # ── setup drawing ────────────────────────────────────────

    def _draw_setup(self):
        self._draw_btn(10, 8, 90, 28,
                       f"Team: {team_name(self.sel_team)}",
                       team_color(self.sel_team), config.WHITE)
        txt = FONT_MD.render("Click palette → click hex. Right-click to remove.", True, config.GRAY)
        self.canvas.blit(txt, (220, 12))

        FONT_MD.set_bold(True)
        self.canvas.blit(FONT_MD.render("UNITS", True, config.GRAY), (60, 48))
        FONT_MD.set_bold(False)
        for i, name in enumerate(config.UNIT_TYPES):
            r = self._palette_rect(i)
            sel = self.sel_type == name
            bg = (60, 60, 80) if sel else config.DARK
            pygame.draw.rect(self.canvas, bg, r, border_radius=4)
            if sel:
                pygame.draw.rect(self.canvas, config.YELLOW, r, 2, border_radius=4)
            ut = config.UNIT_TYPES[name]
            tc = team_light(self.sel_team)
            self.canvas.blit(FONT_MD.render(ut["symbol"], True, tc), (r.x + 6, r.y + 4))
            self.canvas.blit(FONT_SM.render(name, True, config.WHITE), (r.x + 24, r.y + 3))
            self.canvas.blit(FONT_SM.render(
                f"A{ut['attack']} D{ut['defense']} H{ut['hp']} S{ut['speed']} x{ut['count']}",
                True, config.GRAY), (r.x + 24, r.y + 20))

        highlights = {}
        if self.hover and self.sel_type:
            if self.grid.half_of(self.hover[0]) == self.sel_team:
                highlights[self.hover] = team_color(self.sel_team)
        self.grid.draw_grid(self.canvas, highlights)
        self._draw_units(self.units)

        can = self._can_start()
        sr = self._start_btn_rect()
        self._draw_btn(sr.x, sr.y, sr.w, sr.h, "Start Battle",
                       config.GREEN if can else config.GRAY, config.BLACK)

        for i, pname in enumerate(config.PRESETS):
            r = self._preset_rect(i)
            pygame.draw.rect(self.canvas, config.DARK, r, border_radius=3)
            pygame.draw.rect(self.canvas, config.GRAY, r, 1, border_radius=3)
            self.canvas.blit(FONT_SM.render(f"Preset: {pname}", True, config.WHITE), (r.x + 6, r.y + 4))

        for team in (0, 1):
            n = sum(1 for u in self.units if u.team == team)
            self.canvas.blit(FONT_SM.render(f"{team_name(team)}: {n} units", True, team_light(team)),
                             (self.grid.ox + (0 if team == 0 else 320), 52))

    def _draw_units(self, units, current=None):
        for u in units:
            if not u.is_alive:
                continue
            cx, cy = self.grid.center(*u.pos)
            color = team_color(u.team)
            if u.is_archer:
                pts = [(cx, cy - 14), (cx - 12, cy + 10), (cx + 12, cy + 10)]
                pygame.draw.polygon(self.canvas, color, pts)
                pygame.draw.polygon(self.canvas, config.WHITE, pts, 1)
            elif u.is_flying:
                pts = [(cx, cy - 14), (cx + 12, cy), (cx, cy + 14), (cx - 12, cy)]
                pygame.draw.polygon(self.canvas, color, pts)
                pygame.draw.polygon(self.canvas, config.WHITE, pts, 1)
            else:
                pygame.draw.circle(self.canvas, color, (int(cx), int(cy)), 13)
                pygame.draw.circle(self.canvas, config.WHITE, (int(cx), int(cy)), 13, 1)
            sym = FONT_MD.render(u.symbol, True, config.WHITE)
            self.canvas.blit(sym, sym.get_rect(center=(cx, cy - 1)))
            hp_ratio = u._total_hp / (u.max_hp * max(u.count, 1))
            bw = 22; bx = cx - bw // 2; by = cy + 16
            pygame.draw.rect(self.canvas, (60, 20, 20), (bx, by, bw, 4))
            c = config.GREEN if hp_ratio > 0.5 else (config.YELLOW if hp_ratio > 0.25 else config.RED)
            pygame.draw.rect(self.canvas, c, (bx, by, max(1, int(bw * hp_ratio)), 4))
            self.canvas.blit(FONT_SM.render(str(u.count), True, config.WHITE), (cx + 14, cy + 8))
            if u is current:
                pygame.draw.circle(self.canvas, config.YELLOW, (int(cx), int(cy)), 18, 2)

    # ── battle drawing ───────────────────────────────────────

    def _draw_battle(self):
        if self.battle:
            for team in (0, 1):
                units = self.battle.alive(team)
                total = sum(u.strength for u in units)
                n = len(units)
                txt = FONT_MD.render(f"{team_name(team)}: {n} units (⚔{total:.0f})", True, team_light(team))
                self.canvas.blit(txt, (self.grid.ox + (0 if team == 0 else 340), 10))
            self.canvas.blit(FONT_MD.render(f"Round {self.battle.round_num}", True, config.WHITE),
                             (self.grid.ox + 180, 10))

        highlights = {}
        if self.b_path:
            for p in self.b_path:
                highlights.setdefault(p, config.PATH_COLOR)
        if self.b_action and isinstance(self.b_action, MoveAction):
            for p in self.b_action.path:
                highlights[p] = tuple(min(highlights.get(p, config.BG)[i] + 40, 255) for i in range(3))
        self.grid.draw_grid(self.canvas, highlights)

        if self.b_target and self.b_action and isinstance(self.b_action, AttackAction):
            self.grid.draw_dashed_line(self.canvas, self.b_action.attacker.pos,
                                       self.b_target.pos, config.TARGET_COLOR, 2)
            self.grid.draw_overlay(self.canvas, self.b_target.pos, config.TARGET_COLOR, 3)

        current_unit = None
        if self.b_action:
            if isinstance(self.b_action, MoveAction): current_unit = self.b_action.unit
            elif isinstance(self.b_action, AttackAction): current_unit = self.b_action.attacker
            elif isinstance(self.b_action, SkipAction): current_unit = self.b_action.unit
        self._draw_units(self.units, current=current_unit)

        if self.debug and self.b_desc:
            bar = pygame.Rect(0, self.bottom_y, VW, VH - self.bottom_y)
            pygame.draw.rect(self.canvas, (20, 20, 28), bar)
            pygame.draw.line(self.canvas, config.GRAY, bar.topleft, bar.topright, 1)
            for i, line in enumerate(self.b_desc.split(" → ")):
                last = (i == len(self.b_desc.split(" → ")) - 1)
                self.canvas.blit(FONT_MD.render(line, True, config.CYAN if last else config.WHITE),
                                 (10, self.bottom_y + 5 + i * 18))
            for i, log in enumerate(self.b_log):
                self.canvas.blit(FONT_SM.render(log, True, config.GRAY),
                                 (VW // 2, self.bottom_y + 5 + i * 14))

        spd_names = ["Slow", "Normal", "Fast"]
        hints = (f"[Space] {'▶ Play' if self.paused else '⏸ Pause'}   "
                 f"[1/2/3] Speed: {spd_names[self.speed]}   "
                 f"[R] Reset   [D] Debug: {'ON' if self.debug else 'OFF'}   "
                 f"[+/-] Size  [F11] Fullscreen")
        self.canvas.blit(FONT_SM.render(hints, True, config.GRAY), (10, VH - 16))

    # ── game over ────────────────────────────────────────────

    def _draw_gameover(self):
        self._draw_battle()
        overlay = pygame.Surface((VW, VH), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.canvas.blit(overlay, (0, 0))
        if self.battle:
            w = self.battle.winner()
            txt = FONT_XL.render(f"{team_name(w)} Wins!", True, team_color(w))
            self.canvas.blit(txt, txt.get_rect(center=(VW // 2, VH // 2 - 20)))
        r = self._playagain_rect()
        self._draw_btn(r.x, r.y, r.w, r.h, "Play Again", config.GREEN, config.BLACK)

    # ── game logic ───────────────────────────────────────────

    def _can_start(self):
        return any(u.team == 0 for u in self.units) and any(u.team == 1 for u in self.units)

    def _start_battle(self):
        self.battle = BattleState(self.grid, self.units)
        self.state = BATTLE
        self.b_sub = B_IDLE; self.b_timer = 0; self.b_log = []
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


if __name__ == "__main__":
    Game().run()
