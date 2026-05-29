"""Battle AI Demo — unified entry point.

Usage:
    battle-ai-demo                       # GUI mode
    battle-ai-demo configs/example.json  # headless CLI
    battle-ai-demo configs/*.json -o out # headless CLI with output
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="HoMM2 Battle AI Demo",
        usage="%(prog)s [config.json ...] [-o output.log]",
        epilog="""examples:
  %(prog)s                          GUI mode (no args)
  %(prog)s configs/example.json     single battle, log auto-named
  %(prog)s configs/a.json -o a.log  single battle, custom output
  %(prog)s configs/*.json           batch mode, each gets its own log""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="Battle config JSON file(s). "
                        "Omit to launch GUI.")
    parser.add_argument("-o", "--output", help="Output log path (single config only)")
    args = parser.parse_args()

    # pygame must init before any engine/ai import (hex_grid imports pygame)
    import pygame
    pygame.init()

    if args.configs:
        # headless CLI mode
        from headless import run_battle

        if args.output and len(args.configs) > 1:
            print("Error: -o can only be used with a single config file",
                  file=sys.stderr)
            sys.exit(1)

        for path in args.configs:
            if not os.path.isfile(path):
                print(f"Config not found: {path}", file=sys.stderr)
                sys.exit(1)
            log = run_battle(path, args.output if len(args.configs) == 1 else None)
            print(f"[{os.path.basename(path)}] -> {log}")
    else:
        # GUI mode
        from ui.game import Game
        Game().run()


if __name__ == "__main__":
    main()
