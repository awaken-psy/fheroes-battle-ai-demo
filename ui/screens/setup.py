"""Setup screen: unit placement, preset loading, team selection."""

import pygame

import config
from .. import fonts
from ..renderer import draw_btn, draw_unit
from engine.unit import Unit


class SetupScreen:
    """Handles the pre-battle setup phase."""

    # Layout constants (in design pixels, before scaling)
    PANEL_X = 4
    PANEL_W = 216
    HEADER_Y = 48
    UNIT_AREA_Y = 64
    UNIT_ROW_H = 50       # single-line row height
    UNIT_ROW_H_TALL = 64  # two-line row height
    UNIT_ROW_GAP = 4
    UNIT_VISIBLE = 6
    PRESET_HEADER_Y = 460
    PRESET_AREA_Y = 476
    PRESET_ROW_H = 28
    PRESET_ROW_H_TALL = 44
    PRESET_ROW_GAP = 3
    PRESET_VISIBLE = 5

    def __init__(self, game):
        self.game = game
        self.sel_type: str | None = None
        self.sel_team = 0
        self.hover: tuple | None = None
        self.unit_scroll = 0
        self.preset_scroll = 0
        self._unit_sb_drag = False
        self._preset_sb_drag = False
        # AI strategy per team: "classic" or "deep"
        self.ai_strategy = {0: "classic", 1: "classic"}
        self._ai_popup_team = None  # which team's popup is open

    # ── name wrapping check ────────────────────────────────────

    def _unit_needs_wrap(self, name):
        """Check if unit name is too wide for one line at current scale."""
        s = self.game._s
        name_surf = fonts.BODY.render(name, True, config.WHITE)
        max_w = s(196 - 34)  # rect width minus symbol column
        return name_surf.get_width() > max_w

    def _preset_needs_wrap(self, label):
        """Check if preset label is too wide for one line."""
        s = self.game._s
        surf = fonts.BODY.render(label, True, config.WHITE)
        max_w = s(196 - 12)
        return surf.get_width() > max_w

    def _unit_row_h(self, item_idx):
        """Height of a specific unit row (design pixels)."""
        names = list(config.UNIT_TYPES)
        if item_idx >= len(names):
            return self.UNIT_ROW_H
        return self.UNIT_ROW_H_TALL if self._unit_needs_wrap(names[item_idx]) else self.UNIT_ROW_H

    def _preset_row_h(self, item_idx):
        """Height of a specific preset row (design pixels)."""
        names = list(config.PRESETS)
        if item_idx >= len(names):
            return self.PRESET_ROW_H
        label = f"Preset: {names[item_idx]}"
        return self.PRESET_ROW_H_TALL if self._preset_needs_wrap(label) else self.PRESET_ROW_H

    # ── layout rects (design pixels → scaled) ─────────────────

    def _unit_y_offset(self, slot):
        """Cumulative Y (design px) for visible unit row `slot`."""
        y = self.UNIT_AREA_Y
        for i in range(slot):
            y += self._unit_row_h(self.unit_scroll + i) + self.UNIT_ROW_GAP
        return y

    def _unit_rect(self, slot):
        """Screen rect for visible unit row `slot`."""
        s = self.game._s
        y = self._unit_y_offset(slot)
        h = self._unit_row_h(self.unit_scroll + slot)
        return pygame.Rect(int(s(12)), int(s(y)), int(s(196)), int(s(h)))

    def _unit_visible_height(self):
        """Total design-pixel height of visible unit rows + gaps."""
        h = 0
        for i in range(self.UNIT_VISIBLE):
            idx = self.unit_scroll + i
            if idx >= len(config.UNIT_TYPES):
                break
            h += self._unit_row_h(idx) + self.UNIT_ROW_GAP
        return h - self.UNIT_ROW_GAP

    def _unit_max_scroll(self):
        return max(0, len(config.UNIT_TYPES) - self.UNIT_VISIBLE)

    def _unit_scrollbar_track(self):
        s = self.game._s
        h = self._unit_visible_height()
        return pygame.Rect(int(s(212)), int(s(self.UNIT_AREA_Y)), int(s(7)), int(s(h)))

    def _unit_scrollbar_thumb(self):
        track = self._unit_scrollbar_track()
        n = len(config.UNIT_TYPES)
        th = max(int(track.h * self.UNIT_VISIBLE / n), int(self.game._s(20)))
        ms = self._unit_max_scroll()
        off = int((track.h - th) * (self.unit_scroll / ms)) if ms else 0
        return pygame.Rect(track.x, track.y + off, track.w, th)

    def _preset_y_offset(self, slot):
        """Cumulative Y (design px) for visible preset row `slot`."""
        y = self.PRESET_AREA_Y
        for i in range(slot):
            y += self._preset_row_h(self.preset_scroll + i) + self.PRESET_ROW_GAP
        return y

    def _preset_rect(self, slot):
        """Screen rect for visible preset row `slot`."""
        s = self.game._s
        y = self._preset_y_offset(slot)
        h = self._preset_row_h(self.preset_scroll + slot)
        return pygame.Rect(int(s(12)), int(s(y)), int(s(196)), int(s(h)))

    def _preset_max_scroll(self):
        return max(0, len(config.PRESETS) - self.PRESET_VISIBLE)

    def _preset_scrollbar_track(self):
        s = self.game._s
        h = 0
        for i in range(self.PRESET_VISIBLE):
            idx = self.preset_scroll + i
            if idx >= len(config.PRESETS):
                break
            h += self._preset_row_h(idx) + self.PRESET_ROW_GAP
        h = max(h - self.PRESET_ROW_GAP, 0)
        return pygame.Rect(int(s(212)), int(s(self.PRESET_AREA_Y)), int(s(7)), int(s(h)))

    def _preset_scrollbar_thumb(self):
        track = self._preset_scrollbar_track()
        n = len(config.PRESETS)
        th = max(int(track.h * self.PRESET_VISIBLE / n), int(self.game._s(20)))
        ms = self._preset_max_scroll()
        off = int((track.h - th) * (self.preset_scroll / ms)) if ms else 0
        return pygame.Rect(track.x, track.y + off, track.w, th)

    def _start_btn_rect(self):
        s = self.game._s
        cx = s(config.WINDOW_WIDTH) // 2
        return pygame.Rect(cx - s(100), s(config.WINDOW_HEIGHT - 55), s(200), s(44))

    def _ai_btn_rect(self, team):
        """Rect for the AI strategy button on the right side, above Start."""
        s = self.game._s
        vw = config.WINDOW_WIDTH
        vh = config.WINDOW_HEIGHT
        w = s(140)
        y = s(vh - 55 - 40 - 10)  # above Start button
        if team == 0:
            x = s(vw) // 2 - w - s(5)
        else:
            x = s(vw) // 2 + s(5)
        return pygame.Rect(int(x), int(y), int(w), int(s(32)))

    def _ai_popup_rect(self):
        """Rect for the AI strategy popup."""
        s = self.game._s
        vw = config.WINDOW_WIDTH
        vh = config.WINDOW_HEIGHT
        w = s(200)
        h = s(100)
        x = s(vw) // 2 - w // 2
        y = s(vh) // 2 - h // 2
        return pygame.Rect(int(x), int(y), int(w), int(h))

    # ── scrollbar helpers ─────────────────────────────────────

    def _scroll_unit_to_pixel(self, my):
        track = self._unit_scrollbar_track()
        ms = self._unit_max_scroll()
        if ms <= 0:
            return
        frac = (my - track.y) / max(track.h, 1)
        self.unit_scroll = max(0, min(ms, round(frac * ms)))

    def _scroll_preset_to_pixel(self, my):
        track = self._preset_scrollbar_track()
        ms = self._preset_max_scroll()
        if ms <= 0:
            return
        frac = (my - track.y) / max(track.h, 1)
        self.preset_scroll = max(0, min(ms, round(frac * ms)))

    # ── AI popup ──────────────────────────────────────────────

    def _handle_ai_popup_click(self, mx, my):
        """Handle clicks inside the AI strategy popup."""
        s = self.game._s
        popup = self._ai_popup_rect()
        # Option rects: Classic / Deep / Cancel
        opts = [("Classic AI", "classic"), ("Deep Learning", "deep")]
        for i, (label, key) in enumerate(opts):
            opt_y = popup.y + s(36 + i * 28)
            opt_rect = pygame.Rect(popup.x + s(10), opt_y,
                                   popup.w - s(20), s(24))
            if opt_rect.collidepoint(mx, my):
                self.ai_strategy[self._ai_popup_team] = key
                self._ai_popup_team = None
                return
        # Cancel (click outside options closes)
        if not popup.collidepoint(mx, my):
            self._ai_popup_team = None

    def _draw_ai_btn(self, canvas, s, team):
        """Draw an AI strategy button. Gray/disabled if player controls this team."""
        r = self._ai_btn_rect(team)
        strategy = self.ai_strategy[team]
        is_player = (self.game.player_team is not None and self.game.player_team == team)
        if is_player:
            label = f"{fonts.team_name(team)}: Player"
            draw_btn(canvas, r.x, r.y, r.w, r.h, label, (60, 60, 60), (120, 120, 120))
        else:
            label = f"{fonts.team_name(team)}: {strategy}"
            bg = fonts.team_color(team)
            draw_btn(canvas, r.x, r.y, r.w, r.h, label, bg, config.BLACK)

    def _draw_ai_popup(self, canvas, s):
        """Draw the AI strategy selection popup."""
        popup = self._ai_popup_rect()
        # Overlay
        overlay = pygame.Surface((self.game.win_w, self.game.win_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.game.canvas.blit(overlay, (0, 0))
        # Panel
        pygame.draw.rect(canvas, (30, 38, 55), popup, border_radius=int(s(6)))
        pygame.draw.rect(canvas, (80, 100, 140), popup, 2, border_radius=int(s(6)))
        # Title
        title = f"AI Strategy: {fonts.team_name(self._ai_popup_team)}"
        canvas.blit(fonts.TITLE.render(title, True, config.WHITE),
                    (popup.x + s(10), popup.y + s(8)))
        # Options
        opts = [("Classic AI", "classic"), ("Deep Learning", "deep")]
        for i, (label, key) in enumerate(opts):
            y = popup.y + s(36 + i * 28)
            sel = self.ai_strategy[self._ai_popup_team] == key
            bg = (50, 55, 78) if sel else (38, 45, 65)
            r = pygame.Rect(popup.x + s(10), y, popup.w - s(20), s(24))
            pygame.draw.rect(canvas, bg, r, border_radius=int(s(3)))
            if sel:
                pygame.draw.rect(canvas, config.YELLOW, r, 2, border_radius=int(s(3)))
            else:
                pygame.draw.rect(canvas, (60, 68, 92), r, 1, border_radius=int(s(3)))
            canvas.blit(fonts.BODY.render(label, True, config.WHITE),
                        (r.x + s(8), r.y + s(3)))

    # ── event handling ────────────────────────────────────────

    def handle(self, ev):
        if ev.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            s = self.game._s
            unit_bottom = s(self.UNIT_AREA_Y + self._unit_visible_height())
            if my < unit_bottom:
                self.unit_scroll = max(0, min(self._unit_max_scroll(),
                                             self.unit_scroll - ev.y))
            else:
                self.preset_scroll = max(0, min(self._preset_max_scroll(),
                                                self.preset_scroll - ev.y))
            return
        if ev.type == pygame.MOUSEBUTTONUP:
            self._unit_sb_drag = False
            self._preset_sb_drag = False
            return
        if ev.type == pygame.MOUSEMOTION:
            if self._unit_sb_drag:
                self._scroll_unit_to_pixel(ev.pos[1]); return
            if self._preset_sb_drag:
                self._scroll_preset_to_pixel(ev.pos[1]); return
            self.hover = self.game.hex_renderer.pixel_to_hex(*ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            s = self.game._s
            if ev.button == 1:
                # AI popup — handle first (highest priority)
                if self._ai_popup_team is not None:
                    self._handle_ai_popup_click(mx, my)
                    return
                # AI strategy buttons
                for team in (0, 1):
                    btn = self._ai_btn_rect(team)
                    if btn.collidepoint(mx, my):
                        # Disabled if player controls this team
                        if self.game.player_team is not None and self.game.player_team == team:
                            return  # ignore click
                        self._ai_popup_team = team
                        return
                # Unit scrollbar
                if self._unit_max_scroll() > 0:
                    if self._unit_scrollbar_thumb().collidepoint(mx, my):
                        self._unit_sb_drag = True; return
                    if self._unit_scrollbar_track().collidepoint(mx, my):
                        self._scroll_unit_to_pixel(my); return
                # Preset scrollbar
                if self._preset_max_scroll() > 0:
                    if self._preset_scrollbar_thumb().collidepoint(mx, my):
                        self._preset_sb_drag = True; return
                    if self._preset_scrollbar_track().collidepoint(mx, my):
                        self._scroll_preset_to_pixel(my); return
                # Unit rows
                names = list(config.UNIT_TYPES)
                for slot in range(min(self.UNIT_VISIBLE, len(names))):
                    if self._unit_rect(slot).collidepoint(mx, my):
                        self.sel_type = names[self.unit_scroll + slot]; return
                # Team / player buttons
                team_rect = pygame.Rect(int(s(14)), int(s(12)),
                                        int(s(100)), int(s(32)))
                if team_rect.collidepoint(mx, my):
                    self.sel_team = 1 - self.sel_team; return
                player_rect = pygame.Rect(int(s(120)), int(s(12)),
                                          int(s(100)), int(s(32)))
                if player_rect.collidepoint(mx, my):
                    if self.game.player_team is None:
                        self.game.player_team = 0
                    elif self.game.player_team == 0:
                        self.game.player_team = 1
                    else:
                        self.game.player_team = None
                    return
                # Preset rows
                preset_names = list(config.PRESETS)
                for slot in range(min(self.PRESET_VISIBLE, len(preset_names))):
                    if self._preset_rect(slot).collidepoint(mx, my):
                        self._load_preset(preset_names[self.preset_scroll + slot]); return
                # Start button
                if self._start_btn_rect().collidepoint(mx, my):
                    if self._can_start():
                        self.game.start_battle()
                    return
                # Hex grid placement
                hex_pos = self.game.hex_renderer.pixel_to_hex(mx, my)
                if hex_pos and self.sel_type:
                    col, row = hex_pos
                    if self.game.grid.half_of(col) == self.sel_team:
                        self.game.units = [u for u in self.game.units if u.pos != hex_pos]
                        self.game.units.append(
                            Unit.from_type(self.sel_type, self.sel_team, col, row))
            elif ev.button == 3:
                hex_pos = self.game.hex_renderer.pixel_to_hex(mx, my)
                if hex_pos:
                    self.game.units = [u for u in self.game.units if u.pos != hex_pos]

    # ── drawing ───────────────────────────────────────────────

    def draw(self):
        g = self.game
        s = g._s
        canvas = g.canvas

        # Reposition grid to centre in available space after left panel
        panel_right = s(228)
        grid_w = g.grid.cols * g.hex_renderer.hex_w
        avail = g.win_w - panel_right
        g.hex_renderer.reposition(panel_right + (avail - grid_w) / 2, s(config.GRID_OFFSET_Y))

        # left panel background
        panel = pygame.Rect(int(s(self.PANEL_X)), int(s(4)), int(s(self.PANEL_W)),
                            int(s(config.WINDOW_HEIGHT) - s(8)))
        pygame.draw.rect(canvas, config.PANEL_BG, panel, border_radius=int(s(6)))
        pygame.draw.rect(canvas, (55, 65, 90), panel, 1, border_radius=int(s(6)))

        # team selector
        draw_btn(canvas, s(14), s(12), s(100), s(32),
                 f"Team: {fonts.team_name(self.sel_team)}",
                 fonts.team_color(self.sel_team), config.WHITE)
        # player side selector (3-state: Blue / Red / Auto)
        if self.game.player_team is None:
            play_label = "Auto Battle"
            play_color = config.GRAY
            play_text_color = config.WHITE
        else:
            play_label = f"Play: {fonts.team_name(self.game.player_team)}"
            play_color = fonts.team_color(self.game.player_team)
            play_text_color = config.BLACK
        draw_btn(canvas, s(120), s(12), s(100), s(32),
                 play_label, play_color, play_text_color)
        hint = fonts.BODY.render("Click palette -> hex. Right-click remove.",
                                 True, (170, 180, 200))
        canvas.blit(hint, (s(230), s(16)))

        # ── UNITS section ──
        canvas.blit(fonts.TITLE.render("UNITS", True, config.WHITE),
                    (s(18), s(self.HEADER_Y)))

        names = list(config.UNIT_TYPES)
        for slot in range(min(self.UNIT_VISIBLE, len(names))):
            name = names[self.unit_scroll + slot]
            r = self._unit_rect(slot)
            sel = self.sel_type == name
            bg = (50, 55, 78) if sel else (38, 45, 65)
            pygame.draw.rect(canvas, bg, r, border_radius=int(s(4)))
            if sel:
                pygame.draw.rect(canvas, config.YELLOW, r, 2, border_radius=int(s(4)))
            else:
                pygame.draw.rect(canvas, (60, 68, 92), r, 1, border_radius=int(s(4)))
            ut = config.UNIT_TYPES[name]
            tc = fonts.team_light(self.sel_team)
            canvas.blit(fonts.LABEL.render(ut["symbol"], True, tc),
                        (r.x + s(8), r.y + s(4)))
            name_surf = fonts.BODY.render(name, True, config.WHITE)
            max_w = r.w - s(34)
            if name_surf.get_width() <= max_w:
                canvas.blit(name_surf, (r.x + s(28), r.y + s(3)))
                canvas.blit(fonts.DATA.render(
                    f"A{ut['attack']} D{ut['defense']} H{ut['hp']} S{ut['speed']} x{ut['count']}",
                    True, (170, 180, 200)), (r.x + s(28), r.y + s(22)))
            else:
                mid = len(name) // 2
                split_idx = name.rfind(' ', 0, mid + 3)
                if split_idx <= 0:
                    split_idx = mid
                line1 = name[:split_idx]
                line2 = name[split_idx + 1:] if split_idx < len(name) else ""
                canvas.blit(fonts.BODY.render(line1, True, config.WHITE),
                            (r.x + s(28), r.y + s(1)))
                canvas.blit(fonts.BODY.render(line2, True, config.WHITE),
                            (r.x + s(28), r.y + s(16)))
                canvas.blit(fonts.DATA.render(
                    f"A{ut['attack']} D{ut['defense']} H{ut['hp']} S{ut['speed']} x{ut['count']}",
                    True, (170, 180, 200)), (r.x + s(28), r.y + s(31)))

        # Unit scrollbar
        if self._unit_max_scroll() > 0:
            track = self._unit_scrollbar_track()
            thumb = self._unit_scrollbar_thumb()
            pygame.draw.rect(canvas, (28, 34, 50), track, border_radius=int(s(3)))
            pygame.draw.rect(canvas, (95, 108, 140), thumb, border_radius=int(s(3)))

        # ── Separator ──
        sep_y = s(self.PRESET_HEADER_Y - 8)
        pygame.draw.line(canvas, (55, 65, 90),
                         (s(14), sep_y), (s(210), sep_y), 1)

        # ── PRESETS section ──
        canvas.blit(fonts.LABEL.render("PRESETS", True, config.WHITE),
                    (s(18), s(self.PRESET_HEADER_Y)))

        preset_names = list(config.PRESETS)
        for slot in range(min(self.PRESET_VISIBLE, len(preset_names))):
            pname = preset_names[self.preset_scroll + slot]
            r = self._preset_rect(slot)
            pygame.draw.rect(canvas, (30, 38, 55), r, border_radius=int(s(3)))
            pygame.draw.rect(canvas, (60, 68, 92), r, 1, border_radius=int(s(3)))
            label = f"Preset: {pname}"
            label_surf = fonts.BODY.render(label, True, config.WHITE)
            max_w = r.w - s(12)
            if label_surf.get_width() <= max_w:
                canvas.blit(label_surf, (r.x + s(8), r.y + s(4)))
            else:
                mid = len(label) // 2
                split_idx = label.rfind(' ', 0, mid + 3)
                if split_idx <= 0:
                    split_idx = mid
                line1 = label[:split_idx]
                line2 = label[split_idx + 1:] if split_idx < len(label) else ""
                canvas.blit(fonts.BODY.render(line1, True, config.WHITE),
                            (r.x + s(8), r.y + s(2)))
                canvas.blit(fonts.BODY.render(line2, True, config.WHITE),
                            (r.x + s(8), r.y + s(16)))

        # Preset scrollbar
        if self._preset_max_scroll() > 0:
            track = self._preset_scrollbar_track()
            thumb = self._preset_scrollbar_thumb()
            pygame.draw.rect(canvas, (28, 34, 50), track, border_radius=int(s(3)))
            pygame.draw.rect(canvas, (95, 108, 140), thumb, border_radius=int(s(3)))

        # hex grid
        highlights = {}
        if self.hover and self.sel_type:
            if g.grid.half_of(self.hover[0]) == self.sel_team:
                highlights[self.hover] = fonts.team_color(self.sel_team)
        g.hex_renderer.draw_grid(canvas, highlights)
        self._draw_units()

        # AI strategy buttons (above Start)
        self._draw_ai_btn(canvas, s, 0)
        self._draw_ai_btn(canvas, s, 1)

        # start button
        sr = self._start_btn_rect()
        can = self._can_start()
        draw_btn(canvas, sr.x, sr.y, sr.w, sr.h, "Start Battle",
                 config.GREEN if can else config.GRAY, config.BLACK)

        # AI popup (on top of everything)
        if self._ai_popup_team is not None:
            self._draw_ai_popup(canvas, s)

        # team unit counts above grid
        for team in (0, 1):
            n = sum(1 for u in g.units if u.team == team)
            txt = fonts.BODY.render(f"{fonts.team_name(team)}: {n} units",
                                    True, fonts.team_light(team))
            if team == 0:
                canvas.blit(txt, (int(g.hex_renderer.ox) + int(s(10)), int(s(56))))
            else:
                canvas.blit(txt, (int(g.hex_renderer.ox) + int(grid_w)
                                  - txt.get_width() - int(s(10)), int(s(56))))

    def _draw_units(self):
        g = self.game
        for u in g.units:
            if not u.is_alive:
                continue
            cx, cy = g.hex_renderer.center(*u.pos)
            draw_unit(g.canvas, g._s, g.hex_renderer, u, cx, cy)

    # ── game logic helpers ────────────────────────────────────

    def _can_start(self):
        return (any(u.team == 0 for u in self.game.units)
                and any(u.team == 1 for u in self.game.units))

    def _load_preset(self, name):
        preset = config.PRESETS[name]
        self.game.units = []
        self.game._siege = preset.get("siege", False)
        for team, placements in preset.items():
            if isinstance(team, str):
                continue
            for type_name, col, row in placements:
                self.game.units.append(Unit.from_type(type_name, team, col, row))
