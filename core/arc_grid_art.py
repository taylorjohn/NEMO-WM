"""
NeMo-WM Grid Art Engine
=========================
Draw shapes, letters, symbols, and patterns on ARC-style grids.
Uses discrete programs to generate visual representations from concepts.

This serves two purposes:
  1. ARC SOLVER: If we can DRAW a pattern, we can UNDERSTAND it
  2. COGNITION: Demonstrates program-centric abstraction (Chollet Type 2)

Capabilities:
  - Geometric shapes (line, rect, circle, triangle, cross, diamond, spiral)
  - Letters & digits (5x3 pixel font, all A-Z and 0-9)
  - Pattern generation (checkerboard, stripes, gradient, border, frame)
  - Scene composition (combine multiple elements on one grid)
  - Grid-to-concept recognition (identify drawn shapes)

Usage:
    from arc_grid_art import GridArtist, PixelFont

    artist = GridArtist(height=15, width=15)
    artist.draw_rect(2, 2, 6, 6, color=3)
    artist.draw_circle(7, 7, 4, color=5)
    artist.draw_text("HI", row=1, col=1, color=2)
    grid = artist.get_grid()
"""
import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# PIXEL FONT — 5x3 bitmaps for A-Z and 0-9
# ═══════════════════════════════════════════════════════════

FONT_5x3 = {
    'A': ['111','101','111','101','101'],
    'B': ['110','101','110','101','110'],
    'C': ['111','100','100','100','111'],
    'D': ['110','101','101','101','110'],
    'E': ['111','100','110','100','111'],
    'F': ['111','100','110','100','100'],
    'G': ['111','100','101','101','111'],
    'H': ['101','101','111','101','101'],
    'I': ['111','010','010','010','111'],
    'J': ['111','001','001','101','111'],
    'K': ['101','110','100','110','101'],
    'L': ['100','100','100','100','111'],
    'M': ['101','111','111','101','101'],
    'N': ['101','111','111','111','101'],
    'O': ['111','101','101','101','111'],
    'P': ['111','101','111','100','100'],
    'Q': ['111','101','101','110','011'],
    'R': ['111','101','110','101','101'],
    'S': ['111','100','111','001','111'],
    'T': ['111','010','010','010','010'],
    'U': ['101','101','101','101','111'],
    'V': ['101','101','101','101','010'],
    'W': ['101','101','111','111','101'],
    'X': ['101','101','010','101','101'],
    'Y': ['101','101','010','010','010'],
    'Z': ['111','001','010','100','111'],
    '0': ['111','101','101','101','111'],
    '1': ['010','110','010','010','111'],
    '2': ['111','001','111','100','111'],
    '3': ['111','001','111','001','111'],
    '4': ['101','101','111','001','001'],
    '5': ['111','100','111','001','111'],
    '6': ['111','100','111','101','111'],
    '7': ['111','001','001','001','001'],
    '8': ['111','101','111','101','111'],
    '9': ['111','101','111','001','111'],
    ' ': ['000','000','000','000','000'],
    '.': ['000','000','000','000','010'],
    '!': ['010','010','010','000','010'],
    '-': ['000','000','111','000','000'],
    '+': ['000','010','111','010','000'],
    '=': ['000','111','000','111','000'],
}


class PixelFont:
    """Render text using 5x3 pixel bitmaps."""

    @staticmethod
    def render_char(ch, color=1):
        """Render a single character as a 5x3 numpy array."""
        bitmap = FONT_5x3.get(ch.upper(), FONT_5x3[' '])
        grid = np.zeros((5, 3), dtype=int)
        for r, row in enumerate(bitmap):
            for c, pixel in enumerate(row):
                if pixel == '1':
                    grid[r, c] = color
        return grid

    @staticmethod
    def render_text(text, color=1, spacing=1):
        """Render a string as a grid. Returns numpy array."""
        chars = [PixelFont.render_char(ch, color) for ch in text]
        if not chars:
            return np.zeros((5, 1), dtype=int)

        total_width = sum(c.shape[1] for c in chars) + spacing * (len(chars) - 1)
        grid = np.zeros((5, total_width), dtype=int)

        col = 0
        for char_grid in chars:
            h, w = char_grid.shape
            grid[:h, col:col+w] = char_grid
            col += w + spacing

        return grid

    @staticmethod
    def render_multiline(lines, color=1, spacing=1, line_spacing=1):
        """Render multiple lines of text."""
        rendered = [PixelFont.render_text(line, color, spacing) for line in lines]
        if not rendered:
            return np.zeros((1, 1), dtype=int)

        max_width = max(r.shape[1] for r in rendered)
        total_height = sum(r.shape[0] for r in rendered) + line_spacing * (len(rendered) - 1)
        grid = np.zeros((total_height, max_width), dtype=int)

        row = 0
        for r in rendered:
            h, w = r.shape
            grid[row:row+h, :w] = r
            row += h + line_spacing

        return grid


# ═══════════════════════════════════════════════════════════
# GRID ARTIST — draw shapes and patterns on grids
# ═══════════════════════════════════════════════════════════

class GridArtist:
    """Draw shapes, patterns, and text on ARC-style grids."""

    def __init__(self, height=15, width=15, bg=0):
        self.grid = np.full((height, width), bg, dtype=int)
        self.h = height
        self.w = width
        self.bg = bg

    def get_grid(self):
        return self.grid.copy()

    def clear(self):
        self.grid[:] = self.bg

    def _set(self, r, c, color):
        if 0 <= r < self.h and 0 <= c < self.w:
            self.grid[r, c] = color

    # ─── BASIC DRAWING ────────────────────────────────────

    def draw_pixel(self, r, c, color=1):
        self._set(r, c, color)

    def draw_line_h(self, r, c1, c2, color=1):
        """Horizontal line."""
        for c in range(min(c1, c2), max(c1, c2) + 1):
            self._set(r, c, color)

    def draw_line_v(self, c, r1, r2, color=1):
        """Vertical line."""
        for r in range(min(r1, r2), max(r1, r2) + 1):
            self._set(r, c, color)

    def draw_line(self, r1, c1, r2, c2, color=1):
        """Bresenham line from (r1,c1) to (r2,c2)."""
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr - dc
        r, c = r1, c1
        while True:
            self._set(r, c, color)
            if r == r2 and c == c2:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    # ─── SHAPES ───────────────────────────────────────────

    def draw_rect(self, r1, c1, r2, c2, color=1, fill=False):
        """Rectangle (outline or filled)."""
        if fill:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    self._set(r, c, color)
        else:
            self.draw_line_h(r1, c1, c2, color)
            self.draw_line_h(r2, c1, c2, color)
            self.draw_line_v(c1, r1, r2, color)
            self.draw_line_v(c2, r1, r2, color)

    def draw_frame(self, r1, c1, r2, c2, color=1, thickness=1):
        """Rectangular frame with thickness."""
        for t in range(thickness):
            self.draw_rect(r1 + t, c1 + t, r2 - t, c2 - t, color)

    def draw_circle(self, cr, cc, radius, color=1, fill=False):
        """Circle using midpoint algorithm."""
        for r in range(self.h):
            for c in range(self.w):
                dist = ((r - cr) ** 2 + (c - cc) ** 2) ** 0.5
                if fill:
                    if dist <= radius:
                        self._set(r, c, color)
                else:
                    if abs(dist - radius) < 0.8:
                        self._set(r, c, color)

    def draw_diamond(self, cr, cc, radius, color=1, fill=False):
        """Diamond (rotated square) using Manhattan distance."""
        for r in range(self.h):
            for c in range(self.w):
                dist = abs(r - cr) + abs(c - cc)
                if fill:
                    if dist <= radius:
                        self._set(r, c, color)
                else:
                    if dist == radius:
                        self._set(r, c, color)

    def draw_triangle(self, r1, c1, r2, c2, r3, c3, color=1):
        """Triangle outline."""
        self.draw_line(r1, c1, r2, c2, color)
        self.draw_line(r2, c2, r3, c3, color)
        self.draw_line(r3, c3, r1, c1, color)

    def draw_cross(self, cr, cc, size, color=1):
        """Plus/cross shape."""
        self.draw_line_h(cr, cc - size, cc + size, color)
        self.draw_line_v(cc, cr - size, cr + size, color)

    def draw_x(self, cr, cc, size, color=1):
        """X shape (diagonal cross)."""
        for i in range(-size, size + 1):
            self._set(cr + i, cc + i, color)
            self._set(cr + i, cc - i, color)

    def draw_arrow(self, r, c, direction='right', size=3, color=1):
        """Arrow in given direction."""
        if direction == 'right':
            self.draw_line_h(r, c, c + size * 2, color)
            self.draw_line(r, c + size * 2, r - size, c + size, color)
            self.draw_line(r, c + size * 2, r + size, c + size, color)
        elif direction == 'down':
            self.draw_line_v(c, r, r + size * 2, color)
            self.draw_line(r + size * 2, c, r + size, c - size, color)
            self.draw_line(r + size * 2, c, r + size, c + size, color)

    def draw_spiral(self, cr, cc, turns=2, color=1):
        """Simple rectangular spiral."""
        r, c = cr, cc
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        length = 1
        d = 0
        steps = 0
        for _ in range(turns * 8 + 4):
            for _ in range(length):
                self._set(r, c, color)
                r += directions[d][0]
                c += directions[d][1]
            d = (d + 1) % 4
            steps += 1
            if steps % 2 == 0:
                length += 1

    # ─── PATTERNS ─────────────────────────────────────────

    def draw_checkerboard(self, r1, c1, r2, c2, color1=1, color2=2, cell_size=1):
        """Checkerboard pattern."""
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if ((r - r1) // cell_size + (c - c1) // cell_size) % 2 == 0:
                    self._set(r, c, color1)
                else:
                    self._set(r, c, color2)

    def draw_stripes_h(self, r1, c1, r2, c2, colors=[1, 2], width=1):
        """Horizontal stripes."""
        for r in range(r1, r2 + 1):
            ci = ((r - r1) // width) % len(colors)
            for c in range(c1, c2 + 1):
                self._set(r, c, colors[ci])

    def draw_stripes_v(self, r1, c1, r2, c2, colors=[1, 2], width=1):
        """Vertical stripes."""
        for c in range(c1, c2 + 1):
            ci = ((c - c1) // width) % len(colors)
            for r in range(r1, r2 + 1):
                self._set(r, c, colors[ci])

    def draw_gradient_h(self, r1, c1, r2, c2, colors=[1, 2, 3, 4]):
        """Horizontal gradient using available colors."""
        w = c2 - c1 + 1
        for c in range(c1, c2 + 1):
            ci = min(int((c - c1) / w * len(colors)), len(colors) - 1)
            for r in range(r1, r2 + 1):
                self._set(r, c, colors[ci])

    def draw_border(self, color=1, thickness=1):
        """Draw border around entire grid."""
        for t in range(thickness):
            self.draw_rect(t, t, self.h - 1 - t, self.w - 1 - t, color)

    def draw_grid_lines(self, spacing=3, color=5):
        """Draw grid lines (dividers)."""
        for r in range(0, self.h, spacing):
            self.draw_line_h(r, 0, self.w - 1, color)
        for c in range(0, self.w, spacing):
            self.draw_line_v(c, 0, self.h - 1, color)

    # ─── TEXT ─────────────────────────────────────────────

    def draw_text(self, text, row=0, col=0, color=1, spacing=1):
        """Draw text using pixel font."""
        rendered = PixelFont.render_text(text, color, spacing)
        h, w = rendered.shape
        for r in range(h):
            for c in range(w):
                if rendered[r, c] != 0:
                    self._set(row + r, col + c, rendered[r, c])

    def draw_char(self, ch, row=0, col=0, color=1):
        """Draw a single character."""
        rendered = PixelFont.render_char(ch, color)
        for r in range(5):
            for c in range(3):
                if rendered[r, c] != 0:
                    self._set(row + r, col + c, rendered[r, c])

    # ─── TRANSFORMS ───────────────────────────────────────

    def stamp(self, pattern, row, col):
        """Stamp a small grid onto this grid at (row, col)."""
        ph, pw = pattern.shape
        for r in range(ph):
            for c in range(pw):
                if pattern[r, c] != 0:
                    self._set(row + r, col + c, int(pattern[r, c]))

    def mirror_h(self):
        """Mirror grid horizontally."""
        self.grid = np.hstack([self.grid, np.fliplr(self.grid)])
        self.w = self.grid.shape[1]

    def mirror_v(self):
        """Mirror grid vertically."""
        self.grid = np.vstack([self.grid, np.flipud(self.grid)])
        self.h = self.grid.shape[0]

    def tile(self, nx, ny):
        """Tile the grid nx times horizontally, ny times vertically."""
        self.grid = np.tile(self.grid, (ny, nx))
        self.h, self.w = self.grid.shape

    def fill_enclosed(self, fill_color=None):
        """Fill enclosed regions."""
        from collections import deque
        if fill_color is None:
            nz = [int(v) for v in self.grid.flatten() if v != self.bg]
            fill_color = max(set(nz), default=1)

        visited = np.zeros((self.h, self.w), dtype=bool)
        q = deque()
        for r in range(self.h):
            for c in [0, self.w - 1]:
                if self.grid[r, c] == self.bg and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))
        for c in range(self.w):
            for r in [0, self.h - 1]:
                if self.grid[r, c] == self.bg and not visited[r, c]:
                    visited[r, c] = True
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.h and 0 <= nc < self.w and not visited[nr, nc] and self.grid[nr, nc] == self.bg:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r, c] == self.bg and not visited[r, c]:
                    self.grid[r, c] = fill_color


# ═══════════════════════════════════════════════════════════
# SHAPE RECOGNIZER — identify shapes in grids (inverse of drawing)
# ═══════════════════════════════════════════════════════════

class ShapeRecognizer:
    """Recognize drawn shapes in ARC grids."""

    @staticmethod
    def identify(grid, bg=0):
        """Identify shapes in a grid. Returns list of descriptions."""
        h, w = grid.shape
        shapes = []

        # Check for rectangle/frame
        nz_rows = [r for r in range(h) if any(grid[r, c] != bg for c in range(w))]
        nz_cols = [c for c in range(w) if any(grid[r, c] != bg for r in range(h))]

        if nz_rows and nz_cols:
            r1, r2 = min(nz_rows), max(nz_rows)
            c1, c2 = min(nz_cols), max(nz_cols)
            bbox_area = (r2 - r1 + 1) * (c2 - c1 + 1)
            nz_count = int(np.sum(grid != bg))

            # Full rectangle?
            if nz_count == bbox_area:
                shapes.append(f'filled_rect({r1},{c1},{r2},{c2})')

            # Frame (hollow rectangle)?
            interior = grid[r1+1:r2, c1+1:c2] if r2 > r1+1 and c2 > c1+1 else np.array([])
            if interior.size > 0 and np.all(interior == bg):
                perimeter = 2 * ((r2-r1+1) + (c2-c1+1)) - 4
                if abs(nz_count - perimeter) <= 2:
                    shapes.append(f'frame({r1},{c1},{r2},{c2})')

        # Check symmetry
        if np.array_equal(grid, np.fliplr(grid)):
            shapes.append('symmetric_h')
        if np.array_equal(grid, np.flipud(grid)):
            shapes.append('symmetric_v')
        if np.array_equal(grid, np.rot90(grid, 2)):
            shapes.append('symmetric_180')

        # Check for cross pattern
        center_r, center_c = h // 2, w // 2
        if grid[center_r, center_c] != bg:
            h_line = all(grid[center_r, c] != bg for c in range(w))
            v_line = all(grid[r, center_c] != bg for r in range(h))
            if h_line and v_line:
                shapes.append(f'cross({center_r},{center_c})')

        # Check for checkerboard
        is_checker = True
        for r in range(h):
            for c in range(w):
                expected = grid[0, 0] if (r + c) % 2 == 0 else grid[0, 1] if w > 1 else grid[1, 0]
                if grid[r, c] != expected:
                    is_checker = False
                    break
            if not is_checker:
                break
        if is_checker and w > 1:
            shapes.append('checkerboard')

        # Check for stripes
        if h > 1:
            all_rows_same = all(
                np.array_equal(grid[r], grid[r])
                for r in range(h)
            )
            row_patterns = set(tuple(grid[r]) for r in range(h))
            if len(row_patterns) <= 2:
                shapes.append(f'h_stripes({len(row_patterns)}_patterns)')

        # Color count
        colors = set(int(v) for v in grid.flatten()) - {bg}
        shapes.append(f'colors={sorted(colors)}')

        return shapes


# ═══════════════════════════════════════════════════════════
# SCENE COMPOSER — build complex scenes from primitives
# ═══════════════════════════════════════════════════════════

def compose_scene(description, size=15):
    """
    Generate a grid from a text description.
    Examples:
      "house" → house shape
      "face" → simple face
      "arrow right" → arrow pointing right
      "border + cross" → border with cross inside
    """
    artist = GridArtist(size, size)

    desc = description.lower().strip()

    if 'house' in desc:
        # Simple house: triangle roof + rectangle body
        mid = size // 2
        artist.draw_rect(size//3, size//4, size-2, size*3//4, color=3, fill=True)
        for i in range(size//3 + 1):
            artist.draw_line_h(size//3 - i, mid - i, mid + i, color=2)
        artist.draw_rect(size*2//3, mid-1, size-2, mid+1, color=4, fill=True)  # door

    elif 'face' in desc:
        mid = size // 2
        artist.draw_circle(mid, mid, size//3, color=3)  # head outline
        artist.draw_pixel(mid-2, mid-2, color=1)  # left eye
        artist.draw_pixel(mid-2, mid+2, color=1)  # right eye
        artist.draw_line_h(mid+2, mid-1, mid+1, color=2)  # mouth

    elif 'arrow' in desc:
        mid = size // 2
        if 'right' in desc:
            artist.draw_arrow(mid, 2, 'right', size//4, color=4)
        elif 'down' in desc:
            artist.draw_arrow(2, mid, 'down', size//4, color=4)

    elif 'tree' in desc:
        mid = size // 2
        artist.draw_line_v(mid, size//2, size-2, color=6)  # trunk
        artist.draw_diamond(size//3, mid, size//4, color=3, fill=True)  # canopy

    elif 'star' in desc:
        mid = size // 2
        artist.draw_x(mid, mid, size//4, color=4)
        artist.draw_cross(mid, mid, size//4, color=4)

    elif 'border' in desc and 'cross' in desc:
        artist.draw_border(color=5)
        mid = size // 2
        artist.draw_cross(mid, mid, mid-1, color=3)

    elif 'checkerboard' in desc:
        artist.draw_checkerboard(0, 0, size-1, size-1, color1=1, color2=2)

    elif 'spiral' in desc:
        artist.draw_spiral(size//2, size//2, turns=2, color=6)

    elif 'text' in desc:
        words = desc.replace('text ', '').upper()
        artist.draw_text(words, row=size//3, col=1, color=2)

    else:
        # Default: draw the description as text
        artist.draw_text(desc[:5].upper(), row=size//3, col=1, color=2)

    return artist.get_grid()


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST & DEMO
# ═══════════════════════════════════════════════════════════

def print_grid(grid, name=""):
    """Pretty-print a grid with colors."""
    color_map = {0:'·',1:'█',2:'▓',3:'▒',4:'░',5:'#',6:'%',7:'@',8:'&',9:'*'}
    if name:
        print(f"\n  {name}:")
    for row in grid:
        print("  " + "".join(color_map.get(int(v), '?') for v in row))


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        NeMo-WM Grid Art Engine                              ║")
    print("║        Drawing shapes, text, and patterns on ARC grids      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Demo 1: Pixel Font
    print("\n" + "═"*60)
    print("  1. PIXEL FONT — 5x3 character rendering")
    print("═"*60)
    for text in ["HELLO", "ARC", "NEMO-WM", "42"]:
        grid = PixelFont.render_text(text)
        print_grid(grid, text)

    # Demo 2: Geometric Shapes
    print("\n" + "═"*60)
    print("  2. GEOMETRIC SHAPES")
    print("═"*60)

    a = GridArtist(9, 9)
    a.draw_rect(1, 1, 7, 7, color=3)
    print_grid(a.get_grid(), "Rectangle (outline)")

    a = GridArtist(9, 9)
    a.draw_circle(4, 4, 3, color=5, fill=True)
    print_grid(a.get_grid(), "Circle (filled)")

    a = GridArtist(9, 9)
    a.draw_diamond(4, 4, 3, color=2)
    print_grid(a.get_grid(), "Diamond")

    a = GridArtist(9, 9)
    a.draw_cross(4, 4, 3, color=4)
    print_grid(a.get_grid(), "Cross")

    a = GridArtist(9, 9)
    a.draw_x(4, 4, 3, color=6)
    print_grid(a.get_grid(), "X shape")

    a = GridArtist(9, 9)
    a.draw_triangle(1, 4, 7, 1, 7, 7, color=3)
    print_grid(a.get_grid(), "Triangle")

    # Demo 3: Patterns
    print("\n" + "═"*60)
    print("  3. PATTERNS")
    print("═"*60)

    a = GridArtist(8, 8)
    a.draw_checkerboard(0, 0, 7, 7, color1=1, color2=2)
    print_grid(a.get_grid(), "Checkerboard")

    a = GridArtist(8, 8)
    a.draw_stripes_h(0, 0, 7, 7, colors=[3, 0, 4], width=1)
    print_grid(a.get_grid(), "Horizontal stripes")

    a = GridArtist(8, 8)
    a.draw_border(color=5, thickness=2)
    print_grid(a.get_grid(), "Double border")

    # Demo 4: Composed Scenes
    print("\n" + "═"*60)
    print("  4. COMPOSED SCENES")
    print("═"*60)

    for scene in ["house", "face", "star", "tree", "border + cross"]:
        grid = compose_scene(scene, size=11)
        print_grid(grid, scene)

    # Demo 5: Text on shapes
    print("\n" + "═"*60)
    print("  5. TEXT ON SHAPES")
    print("═"*60)

    a = GridArtist(11, 19)
    a.draw_rect(0, 0, 10, 18, color=5)
    a.draw_rect(1, 1, 9, 17, color=0, fill=True)
    a.draw_text("NEMO", row=3, col=3, color=3)
    print_grid(a.get_grid(), "Framed text")

    # Demo 6: Shape Recognition
    print("\n" + "═"*60)
    print("  6. SHAPE RECOGNITION (inverse of drawing)")
    print("═"*60)

    test_grids = {
        'frame': lambda: (lambda a: (a.draw_rect(1,1,7,7,color=3), a.get_grid())[-1])(GridArtist(9,9)),
        'filled': lambda: (lambda a: (a.draw_rect(1,1,7,7,color=3,fill=True), a.get_grid())[-1])(GridArtist(9,9)),
        'checker': lambda: (lambda a: (a.draw_checkerboard(0,0,7,7), a.get_grid())[-1])(GridArtist(8,8)),
        'cross': lambda: (lambda a: (a.draw_cross(4,4,3,color=4), a.get_grid())[-1])(GridArtist(9,9)),
    }

    for name, gen_fn in test_grids.items():
        grid = gen_fn()
        shapes = ShapeRecognizer.identify(grid)
        print(f"  {name}: {shapes}")

    print(f"\n  Grid Art Engine: ready")
    print(f"  Characters: {len(FONT_5x3)} (A-Z, 0-9, punctuation)")
    print(f"  Shapes: line, rect, circle, diamond, triangle, cross, X, arrow, spiral")
    print(f"  Patterns: checkerboard, stripes, gradient, border, grid lines")
    print(f"  Scenes: house, face, star, tree, arrow, border+cross")
    print(f"  Recognition: frame, filled, checkerboard, cross, symmetry, stripes")
