import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional

class GridDrawer:
    """
    A class to create and draw a grid of horizontal and vertical sine wave lines.
    Each line completes 3 cycles (3 * 360 degrees) across its length.
    """
    
    def __init__(self, width: float = 10, height: float = 10, 
                 rows: int = 10, cols: int = 10, line_color: str = 'black', 
                 line_width: float = 1.0, wave_amplitude: float = None):
        """
        Initialize the grid drawer.
        
        Args:
            width: Total width of the grid
            height: Total height of the grid
            rows: Number of horizontal lines (grid rows)
            cols: Number of vertical lines (grid columns)
            line_color: Color of the grid lines
            line_width: Width of the grid lines
            wave_amplitude: Amplitude of the sine waves (defaults to 10% of spacing)
        """
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.line_color = line_color
        self.line_width = line_width
        self.wave_amplitude = wave_amplitude
        
    def create_lines(self, num_points: int = 40):
        """Create vertical and horizontal line coordinate arrays.

        Returns two lists: `vertical_lines` and `horizontal_lines`, where each
        item is a tuple `(x_values, y_values)` ready to be plotted.
        """
        # Grid occupies half the canvas, centered
        grid_width = self.width / 2
        grid_height = self.height / 2
        x_offset = self.width / 4  # Center horizontally
        y_offset = self.height / 4  # Center vertically

        # Calculate spacing within the grid area
        x_spacing = grid_width / (self.cols + 1)
        y_spacing = grid_height / (self.rows + 1)

        # Set wave amplitude (default to 10% of the smaller spacing)
        wave_amp = self.wave_amplitude if self.wave_amplitude is not None else min(x_spacing, y_spacing) * 0.2

        # Number of cycles per line
        cycles = 3

        vertical_lines = []
        horizontal_lines = []

        # Create vertical lines (sine waves oscillating horizontally)
        for j in range(1, self.cols + 1):
            x_base = x_offset + j * x_spacing
            y_values = np.linspace(y_offset, y_offset + grid_height, num_points)
            x_values = x_base + wave_amp * np.sin(cycles * 2 * np.pi * (y_values - y_offset) / grid_height)
            vertical_lines.append((x_values, y_values))

        # Create horizontal lines (sine waves oscillating vertically)
        for i in range(1, self.rows + 1):
            y_base = y_offset + i * y_spacing
            x_values = np.linspace(x_offset, x_offset + grid_width, num_points)
            y_values = y_base + wave_amp * np.sin(cycles * 2 * np.pi * (x_values - x_offset) / grid_width)
            horizontal_lines.append((x_values, y_values))

        return vertical_lines, horizontal_lines

    def draw_grid(self, show: bool = True, save_path: Optional[str] = None, num_points: int = 40):
        """Draw the grid using matplotlib. Uses `create_lines` to generate data.

        Adds interactive mouse handlers so a clicked point on any line can be
        dragged with the mouse and the figure updates in real time.
        """
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        # compute grid bounds for boundary drawing
        grid_width = self.width / 2
        grid_height = self.height / 2
        x_offset = self.width / 4
        y_offset = self.height / 4

        vertical_lines, horizontal_lines = self.create_lines(num_points=num_points)

        # Store line data and artist references for interactivity
        self._lines = []  # list of dicts: {'x': np.ndarray, 'y': np.ndarray, 'artist': Line2D}

        # plotting vertical lines
        for x_values, y_values in vertical_lines:
            x_arr = np.asarray(x_values)
            y_arr = np.asarray(y_values)
            (artist,) = ax.plot(x_arr, y_arr, color=self.line_color, linewidth=self.line_width, picker=5)
            self._lines.append({'x': x_arr, 'y': y_arr, 'artist': artist, 'orient': 'v'})

        # plotting horizontal lines
        for x_values, y_values in horizontal_lines:
            x_arr = np.asarray(x_values)
            y_arr = np.asarray(y_values)
            (artist,) = ax.plot(x_arr, y_arr, color=self.line_color, linewidth=self.line_width, picker=5)
            self._lines.append({'x': x_arr, 'y': y_arr, 'artist': artist, 'orient': 'h'})

        # Boundary is built from current line endpoints; maintain as a patch
        self._boundary = None

        def update_boundary():
            # collect horizontals and verticals from stored lines
            horizontals = [info for info in self._lines if info.get('orient') == 'h']
            verticals = [info for info in self._lines if info.get('orient') == 'v']

            pts = []
            # left ends of horizontal lines (bottom -> top)
            for info in horizontals:
                pts.append((float(info['x'][0]), float(info['y'][0])))

            # top ends of vertical lines (left -> right)
            for info in verticals:
                pts.append((float(info['x'][-1]), float(info['y'][-1])))

            # right ends of horizontal lines (top -> bottom)
            for info in reversed(horizontals):
                pts.append((float(info['x'][-1]), float(info['y'][-1])))

            # bottom ends of vertical lines (right -> left)
            for info in reversed(verticals):
                pts.append((float(info['x'][0]), float(info['y'][0])))

            # remove consecutive duplicates
            cleaned = []
            for p in pts:
                if not cleaned or p[0] != cleaned[-1][0] or p[1] != cleaned[-1][1]:
                    cleaned.append(p)
            if cleaned and cleaned[0][0] == cleaned[-1][0] and cleaned[0][1] == cleaned[-1][1]:
                cleaned = cleaned[:-1]

            if len(cleaned) < 3:
                return

            if self._boundary is None:
                self._boundary = patches.Polygon(cleaned, closed=True, linewidth=self.line_width * 1,
                                                  edgecolor=self.line_color, facecolor='none')
                ax.add_patch(self._boundary)
            else:
                self._boundary.set_xy(cleaned)

        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # interactive state
        selected = {'line_idx': None, 'pt_idx': None}

        # tolerance for selecting a point (in data units)
        tol = max(self.width, self.height) * 0.02

        def on_press(event):
            if event.inaxes is not ax:
                return
            xclick, yclick = event.xdata, event.ydata
            best = {'dist': float('inf'), 'line_idx': None, 'pt_idx': None}
            for li, info in enumerate(self._lines):
                dx = info['x'] - xclick
                dy = info['y'] - yclick
                dists = np.hypot(dx, dy)
                idx = int(np.argmin(dists))
                if dists[idx] < best['dist']:
                    best.update({'dist': float(dists[idx]), 'line_idx': li, 'pt_idx': idx})

            if best['dist'] <= tol:
                selected['line_idx'] = best['line_idx']
                selected['pt_idx'] = best['pt_idx']

        def on_motion(event):
            if selected['line_idx'] is None:
                return
            if event.inaxes is not ax:
                return
            li = selected['line_idx']
            pi = selected['pt_idx']
            info = self._lines[li]
            # update the selected point to mouse position
            info['x'][pi] = event.xdata
            info['y'][pi] = event.ydata
            info['artist'].set_data(info['x'], info['y'])
            # update polygon boundary to follow moved endpoints
            try:
                update_boundary()
            except Exception:
                pass
            fig.canvas.draw_idle()

        def on_release(event):
            selected['line_idx'] = None
            selected['pt_idx'] = None

        cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
        cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
        cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
        # draw initial polygon boundary from current lines
        try:
            update_boundary()
        except Exception:
            pass
        
        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grid saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig, ax


def main():
    """
    Main function to demonstrate the grid drawer.
    """
    # Create a grid drawer instance
    # You can customize these parameters:
    # - width, height: Size of the grid
    # - rows, cols: Number of grid cells (which determines line count)
    # - line_color: Color of the lines (e.g., 'black', 'blue', '#FF5733')
    # - line_width: Thickness of the lines
    
    grid = GridDrawer(
        width=6,
        height=6,
        rows=3,  # This creates 3 horizontal lines
        cols=3,  # This creates 3 vertical lines
        line_color='black',
        line_width=1.0
    )
    
    # Draw the grid and save it
    grid.draw_grid(show=True, save_path='grid.png')


if __name__ == '__main__':
    main()
