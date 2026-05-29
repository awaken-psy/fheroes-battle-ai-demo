"""Battle AI Demo — entry point.

Controls:
  Setup:  click palette -> click hex to place, right-click to remove
  Battle: Space=pause/step, 1/2/3=speed, R=reset, D=toggle AI debug
  Window: +/- zoom in/out, F11 toggle fullscreen, drag edge to resize
"""

from ui.game import Game

if __name__ == "__main__":
    Game().run()
