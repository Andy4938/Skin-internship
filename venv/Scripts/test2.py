import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button

class CollagenFiberGridModel:
    def __init__(self):
        self.fibers = []  # List of fibers, each fiber is a list of nodes
        self.connections = []  # All connections across all fibers
        self.base_stiffness = 0.1
        self.max_stiffness = 2.0
        self.iterations = 10  # Reduced from 30 to prevent lag
        self.dragging_info = None  # {'fiber_idx': int, 'node_idx': int}
        self.node_radius = 10
        self.zoom_level = 1.5
        self.base_xlim = 800
        self.base_ylim = 650
        self.physics_enabled = True
        self.damping = 0.65
        self.shape_memory_strength = 0.005
        self.damage_threshold = 0.95
        
        # Grid parameters
        self.num_horizontal_fibers = 3  # Default to 3x3
        self.num_vertical_fibers = 3
        self.num_nodes_per_fiber = 12  # Reduced from 15 for better performance
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.patch.set_facecolor('white')
        plt.subplots_adjust(bottom=0.2, left=0.15)
        
        # Initialize the fiber grid
        self.initialize_fiber_grid()
        
        self.setup_plot()
        self.setup_sliders()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def initialize_fiber_grid(self):
        """Create a grid of horizontal and vertical collagen fibers"""
        self.fibers = []
        self.connections = []
        
        grid_width = 600
        grid_height = 500
        x_offset = 100
        y_offset = 75
        
        # Spacing between fibers
        h_spacing = grid_height / (self.num_horizontal_fibers + 1)
        v_spacing = grid_width / (self.num_vertical_fibers + 1)
        
        # Crimp parameters
        amplitude = 15  # Reduced from 80 for grid
        frequency = 2
        
        fiber_id = 0
        
        # Create horizontal fibers
        horizontal_fibers = []
        for i in range(self.num_horizontal_fibers):
            fiber_nodes = []
            y_base = y_offset + (i + 1) * h_spacing
            
            for j in range(self.num_nodes_per_fiber):
                x = x_offset + (j / (self.num_nodes_per_fiber - 1)) * grid_width
                y = y_base + amplitude * np.sin(frequency * 2 * np.pi * j / (self.num_nodes_per_fiber - 1))
                
                node = {
                    'fiber_id': fiber_id,
                    'node_id': j,
                    'x': x,
                    'y': y,
                    'vx': 0,
                    'vy': 0,
                    'fixed': j == 0,  # Fix only leftmost node
                    'original_x': x,
                    'original_y': y,
                    'h_fiber_idx': i,
                    'v_fiber_idx': None
                }
                fiber_nodes.append(node)
            
            horizontal_fibers.append(fiber_nodes)
            self.fibers.append(fiber_nodes)
            
            # Create connections within this horizontal fiber
            for j in range(self.num_nodes_per_fiber - 1):
                rest_length = self.distance_nodes(fiber_nodes[j], fiber_nodes[j + 1])
                self.connections.append({
                    'fiber_id': fiber_id,
                    'node1': fiber_nodes[j],
                    'node2': fiber_nodes[j + 1],
                    'rest_length': rest_length,
                    'straight_length': rest_length * 1.5,
                    'max_damage': 0.0
                })
            
            fiber_id += 1
        
        # Create vertical fibers
        vertical_fibers = []
        for i in range(self.num_vertical_fibers):
            fiber_nodes = []
            x_base = x_offset + (i + 1) * v_spacing
            
            for j in range(self.num_nodes_per_fiber):
                y = y_offset + (j / (self.num_nodes_per_fiber - 1)) * grid_height
                x = x_base + amplitude * np.sin(frequency * 2 * np.pi * j / (self.num_nodes_per_fiber - 1))
                
                node = {
                    'fiber_id': fiber_id,
                    'node_id': j,
                    'x': x,
                    'y': y,
                    'vx': 0,
                    'vy': 0,
                    'fixed': False,  # Vertical fiber nodes are not fixed
                    'original_x': x,
                    'original_y': y,
                    'h_fiber_idx': None,
                    'v_fiber_idx': i
                }
                fiber_nodes.append(node)
            
            vertical_fibers.append(fiber_nodes)
            self.fibers.append(fiber_nodes)
            
            # Create connections within this vertical fiber
            for j in range(self.num_nodes_per_fiber - 1):
                rest_length = self.distance_nodes(fiber_nodes[j], fiber_nodes[j + 1])
                self.connections.append({
                    'fiber_id': fiber_id,
                    'node1': fiber_nodes[j],
                    'node2': fiber_nodes[j + 1],
                    'rest_length': rest_length,
                    'straight_length': rest_length * 1.5,
                    'max_damage': 0.0
                })
            
            fiber_id += 1
        
        # Create cross-connections at intersection points
        intersection_tolerance = 30
        
        for h_fiber in horizontal_fibers:
            for v_fiber in vertical_fibers:
                for h_node in h_fiber:
                    for v_node in v_fiber:
                        dist = self.distance_nodes(h_node, v_node)
                        if dist < intersection_tolerance:
                            rest_length = dist if dist > 0 else 1.0
                            self.connections.append({
                                'fiber_id': -1,
                                'node1': h_node,
                                'node2': v_node,
                                'rest_length': rest_length,
                                'straight_length': rest_length * 1.5,
                                'max_damage': 0.0,
                                'is_cross_connection': True
                            })
        
        # Create boundary connections
        # Collect all endpoint nodes
        # Left side: first nodes of horizontal fibers (bottom to top)
        left_endpoints = [h_fiber[0] for h_fiber in horizontal_fibers]
        
        # Top side: last nodes of vertical fibers (left to right)
        top_endpoints = [v_fiber[-1] for v_fiber in vertical_fibers]
        
        # Right side: last nodes of horizontal fibers (top to bottom)
        right_endpoints = [h_fiber[-1] for h_fiber in reversed(horizontal_fibers)]
        
        # Bottom side: first nodes of vertical fibers (right to left)
        bottom_endpoints = [v_fiber[0] for v_fiber in reversed(vertical_fibers)]
        
        # Only create boundary connections if we have fibers
        if len(horizontal_fibers) > 0 or len(vertical_fibers) > 0:
            # Connect left endpoints (going upward)
            for i in range(len(left_endpoints) - 1):
                rest_length = self.distance_nodes(left_endpoints[i], left_endpoints[i + 1])
                self.connections.append({
                    'fiber_id': -2,
                    'node1': left_endpoints[i],
                    'node2': left_endpoints[i + 1],
                    'rest_length': rest_length,
                    'straight_length': rest_length * 1.5,
                    'max_damage': 0.0,
                    'is_boundary': True
                })
            
            # Connect top endpoints
            for i in range(len(top_endpoints) - 1):
                rest_length = self.distance_nodes(top_endpoints[i], top_endpoints[i + 1])
                self.connections.append({
                    'fiber_id': -2,
                    'node1': top_endpoints[i],
                    'node2': top_endpoints[i + 1],
                    'rest_length': rest_length,
                    'straight_length': rest_length * 1.5,
                    'max_damage': 0.0,
                    'is_boundary': True
                })
            
            # Connect right endpoints
            for i in range(len(right_endpoints) - 1):
                rest_length = self.distance_nodes(right_endpoints[i], right_endpoints[i + 1])
                self.connections.append({
                    'fiber_id': -2,
                    'node1': right_endpoints[i],
                    'node2': right_endpoints[i + 1],
                    'rest_length': rest_length,
                    'straight_length': rest_length * 1.5,
                    'max_damage': 0.0,
                    'is_boundary': True
                })
            
            # Connect bottom endpoints
            for i in range(len(bottom_endpoints) - 1):
                rest_length = self.distance_nodes(bottom_endpoints[i], bottom_endpoints[i + 1])
                self.connections.append({
                    'fiber_id': -2,
                    'node1': bottom_endpoints[i],
                    'node2': bottom_endpoints[i + 1],
                    'rest_length': rest_length,
                    'straight_length': rest_length * 1.5,
                    'max_damage': 0.0,
                    'is_boundary': True
                })
            
            # Close the boundary corners (only if we have endpoints to connect)
            corner_connections = []
            if len(left_endpoints) > 0 and len(top_endpoints) > 0:
                corner_connections.append((left_endpoints[-1], top_endpoints[0]))
            if len(top_endpoints) > 0 and len(right_endpoints) > 0:
                corner_connections.append((top_endpoints[-1], right_endpoints[0]))
            if len(right_endpoints) > 0 and len(bottom_endpoints) > 0:
                corner_connections.append((right_endpoints[-1], bottom_endpoints[0]))
            if len(bottom_endpoints) > 0 and len(left_endpoints) > 0:
                corner_connections.append((bottom_endpoints[-1], left_endpoints[0]))
            
            for n1, n2 in corner_connections:
                rest_length = self.distance_nodes(n1, n2)
                self.connections.append({
                    'fiber_id': -2,
                    'node1': n1,
                    'node2': n2,
                    'rest_length': rest_length,
                    'straight_length': rest_length * 1.5,
                    'max_damage': 0.0,
                    'is_boundary': True
                })
    
    def distance_nodes(self, n1, n2):
        """Calculate distance between two node dictionaries"""
        return np.sqrt((n2['x'] - n1['x'])**2 + (n2['y'] - n1['y'])**2)
    
    def calculate_straightness(self, conn):
        """Calculate how straight a connection is (0 = crimped, 1 = fully straight)"""
        current_length = self.distance_nodes(conn['node1'], conn['node2'])
        rest_length = conn['rest_length']
        straight_length = conn['straight_length']
        
        if current_length <= rest_length:
            return 0.0
        
        straightness = (current_length - rest_length) / (straight_length - rest_length)
        return np.clip(straightness, 0.0, 1.0)
    
    def get_adaptive_stiffness(self, conn):
        """Calculate stiffness based on how straight the fiber is"""
        straightness = self.calculate_straightness(conn)
        
        if straightness < 0.3:
            return self.base_stiffness
        elif straightness < 0.8:
            progress = (straightness - 0.3) / 0.5
            return self.base_stiffness + (self.max_stiffness - self.base_stiffness) * progress * 0.5
        else:
            progress = (straightness - 0.8) / 0.2
            return self.base_stiffness + (self.max_stiffness - self.base_stiffness) * (0.5 + progress * 0.5)
    
    def simulate_physics(self):
        """Run physics simulation"""
        # Apply spring forces
        for conn in self.connections:
            n1 = conn['node1']
            n2 = conn['node2']
            
            if n1['fixed'] and n2['fixed']:
                continue
            
            dx = n2['x'] - n1['x']
            dy = n2['y'] - n1['y']
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance == 0:
                continue
            
            stiffness = self.get_adaptive_stiffness(conn)
            displacement = distance - conn['rest_length']
            max_displacement = conn['straight_length'] - conn['rest_length']
            displacement = np.clip(displacement, -max_displacement, max_displacement)
            
            force = displacement * stiffness
            fx = (dx / distance) * force
            fy = (dy / distance) * force
            
            if not n1['fixed']:
                n1['vx'] += fx
                n1['vy'] += fy
            
            if not n2['fixed']:
                n2['vx'] -= fx
                n2['vy'] -= fy
        
        # Apply shape memory forces - OPTION 3: Update rest lengths to match new configuration
        if self.dragging_info is None:
            for fiber in self.fibers:
                for node in fiber:
                    if not node['fixed']:
                        dx_orig = node['original_x'] - node['x']
                        dy_orig = node['original_y'] - node['y']
                        
                        # Shape memory always tries to restore original crimped position
                        node['vx'] += dx_orig * self.shape_memory_strength
                        node['vy'] += dy_orig * self.shape_memory_strength
        
        # Update positions with damping
        for fiber in self.fibers:
            for node in fiber:
                if not node['fixed']:
                    node['vx'] *= self.damping
                    node['vy'] *= self.damping
                    
                    max_velocity = 50
                    node['vx'] = np.clip(node['vx'], -max_velocity, max_velocity)
                    node['vy'] = np.clip(node['vy'], -max_velocity, max_velocity)
                    
                    node['x'] += node['vx']
                    node['y'] += node['vy']
    
    def check_for_damage(self):
        """Check all connections for damage - Update entire fiber when any segment is damaged
        
        This function is called once on mouse release. It:
        1. Checks which segments exceeded damage threshold
        2. Identifies which fibers those segments belong to
        3. Updates ALL segments and nodes in those fibers to the current deformed state
        """
        any_damage = False
        damaged_fibers = set()
        
        # First pass: identify which fibers have damage
        # Important: Check straightness at THIS moment (before any updates)
        damage_info = []
        for conn in self.connections:
            straightness = self.calculate_straightness(conn)
            
            if straightness >= self.damage_threshold:
                fiber_id = conn.get('fiber_id', -1)
                if fiber_id >= 0:  # Only for regular fiber connections
                    damaged_fibers.add(fiber_id)
                    damage_info.append({
                        'conn': conn,
                        'straightness': straightness,
                        'fiber_id': fiber_id
                    })
        
        # Second pass: update ALL connections and nodes in damaged fibers
        # This happens atomically - all at once
        if damaged_fibers:
            print(f"  DEBUG: Fibers to damage: {damaged_fibers}")
            any_damage = True
            
            # Update all connections in damaged fibers
            updated_conns = 0
            for conn in self.connections:
                fiber_id = conn.get('fiber_id', -1)
                if fiber_id in damaged_fibers:
                    current_length = self.distance_nodes(conn['node1'], conn['node2'])
                    conn['rest_length'] = current_length
                    conn['straight_length'] = current_length * 1.5
                    conn['max_damage'] = 1.0  # Mark as damaged
                    updated_conns += 1
            
            print(f"  DEBUG: Updated {updated_conns} connections")
            
            # Update all nodes in damaged fibers
            updated_nodes = 0
            for fiber in self.fibers:
                if len(fiber) > 0:
                    fiber_id = fiber[0]['fiber_id']
                    if fiber_id in damaged_fibers:
                        for node in fiber:
                            node['original_x'] = node['x']
                            node['original_y'] = node['y']
                            updated_nodes += 1
            
            print(f"  DEBUG: Updated {updated_nodes} nodes")
        
        return any_damage
    
    def setup_plot(self):
        """Set up the matplotlib plot"""
        xlim = self.base_xlim * self.zoom_level
        ylim = self.base_ylim * self.zoom_level
        
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        self.ax.text(xlim/2, ylim - 30, 'Collagen Fiber Grid - Three-Phase Mechanical Response', 
                    ha='center', va='top', fontsize=14, fontweight='bold')
    
    def setup_sliders(self):
        """Set up control sliders"""
        slider_ax1 = plt.axes([0.25, 0.15, 0.5, 0.02])
        self.slider_h_fibers = Slider(
            ax=slider_ax1,
            label='Horizontal Fibers',
            valmin=0,
            valmax=10,
            valinit=self.num_horizontal_fibers,
            valstep=1,
            color='steelblue'
        )
        self.slider_h_fibers.on_changed(self.update_grid)
        
        slider_ax2 = plt.axes([0.25, 0.12, 0.5, 0.02])
        self.slider_v_fibers = Slider(
            ax=slider_ax2,
            label='Vertical Fibers',
            valmin=0,
            valmax=10,
            valinit=self.num_vertical_fibers,
            valstep=1,
            color='green'
        )
        self.slider_v_fibers.on_changed(self.update_grid)
        
        slider_ax3 = plt.axes([0.25, 0.09, 0.5, 0.02])
        self.slider_nodes = Slider(
            ax=slider_ax3,
            label='Nodes per Fiber',
            valmin=8,
            valmax=20,
            valinit=self.num_nodes_per_fiber,
            valstep=1,
            color='purple'
        )
        self.slider_nodes.on_changed(self.update_grid)
        
        slider_ax4 = plt.axes([0.25, 0.06, 0.5, 0.02])
        self.slider_zoom = Slider(
            ax=slider_ax4,
            label='Zoom Out',
            valmin=1.0,
            valmax=3.0,
            valinit=self.zoom_level,
            valfmt='%.1fx',
            color='teal'
        )
        self.slider_zoom.on_changed(self.update_zoom)
        
        slider_ax5 = plt.axes([0.25, 0.03, 0.5, 0.02])
        self.slider_damping = Slider(
            ax=slider_ax5,
            label='Damping',
            valmin=0.5,
            valmax=0.99,
            valinit=self.damping,
            valfmt='%.2f',
            color='orange'
        )
        self.slider_damping.on_changed(self.update_damping)
        
        slider_ax6 = plt.axes([0.08, 0.25, 0.02, 0.5])
        self.slider_shape_memory = Slider(
            ax=slider_ax6,
            label='Crimp\nMemory',
            valmin=0.0,
            valmax=0.02,
            valinit=self.shape_memory_strength,
            valfmt='%.3f',
            color='darkviolet',
            orientation='vertical'
        )
        self.slider_shape_memory.on_changed(self.update_shape_memory)
        
        button_ax = plt.axes([0.85, 0.12, 0.12, 0.04])
        self.button_physics = Button(button_ax, 'Physics: ON', color='lightgreen')
        self.button_physics.on_clicked(self.toggle_physics)
        
        button_ax_reset = plt.axes([0.85, 0.06, 0.12, 0.04])
        self.button_reset = Button(button_ax_reset, 'Reset', color='lightcoral')
        self.button_reset.on_clicked(self.reset)
    
    def update_grid(self, val):
        self.num_horizontal_fibers = int(self.slider_h_fibers.val)
        self.num_vertical_fibers = int(self.slider_v_fibers.val)
        self.num_nodes_per_fiber = int(self.slider_nodes.val)
        self.dragging_info = None
        self.initialize_fiber_grid()
        self.draw()
    
    def update_zoom(self, val):
        self.zoom_level = val
        self.draw()
    
    def update_damping(self, val):
        self.damping = val
    
    def update_shape_memory(self, val):
        self.shape_memory_strength = val
    
    def toggle_physics(self, event):
        self.physics_enabled = not self.physics_enabled
        if self.physics_enabled:
            self.button_physics.label.set_text('Physics: ON')
            self.button_physics.color = 'lightgreen'
        else:
            self.button_physics.label.set_text('Physics: OFF')
            self.button_physics.color = 'lightcoral'
        self.button_physics.ax.set_facecolor(self.button_physics.color)
        self.fig.canvas.draw()
    
    def reset(self, event):
        self.dragging_info = None
        self.initialize_fiber_grid()
        self.draw()
        print("Grid reset to original undamaged state")
    
    def draw(self):
        """Draw the fiber grid with color coding"""
        self.ax.clear()
        self.setup_plot()
        
        for conn in self.connections:
            n1 = conn['node1']
            n2 = conn['node2']
            
            straightness = self.calculate_straightness(conn)
            
            if straightness < 0.3:
                intensity = straightness / 0.3
                color = (0.2, 0.5 + 0.5*intensity, 1.0)
            elif straightness < 0.8:
                intensity = (straightness - 0.3) / 0.5
                r = 0.2 + 0.8 * intensity
                g = 1.0
                b = 1.0 - intensity
                color = (r, g, b)
            elif straightness < self.damage_threshold:
                intensity = (straightness - 0.8) / (self.damage_threshold - 0.8)
                color = (1.0, 1.0 - 0.5*intensity, 0.0)
            else:
                color = (1.0, 0.0, 0.0)
            
            if conn.get('is_boundary', False):
                self.ax.plot([n1['x'], n2['x']], [n1['y'], n2['y']], 
                            color=color, linewidth=3.0, alpha=0.9, solid_capstyle='round')
            elif conn.get('is_cross_connection', False):
                self.ax.plot([n1['x'], n2['x']], [n1['y'], n2['y']], 
                            color=color, linewidth=1.5, alpha=0.5, linestyle=':', solid_capstyle='round')
            else:
                self.ax.plot([n1['x'], n2['x']], [n1['y'], n2['y']], 
                            color=color, linewidth=2.5, alpha=0.8, solid_capstyle='round')
        
        for fiber in self.fibers:
            for node in fiber:
                if node['fixed']:
                    color = 'darkred'
                    size = self.node_radius * 1.2
                else:
                    color = 'steelblue'
                    size = self.node_radius * 0.7
                
                circle = Circle((node['x'], node['y']), size, 
                              color=color, alpha=0.6, zorder=10, edgecolor='white', linewidth=1)
                self.ax.add_patch(circle)
        
        ylim = self.base_ylim * self.zoom_level
        legend_text = 'Blue/Cyan: Elastic | Yellow: Stiffening | Orange: Very Stiff | RED: DAMAGE!'
        self.ax.text(50, ylim - 40, legend_text, 
                    fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        conn_info_text = 'Thick lines = boundary | Dotted = intersections | Medium = fibers'
        self.ax.text(50, ylim - 65, conn_info_text, 
                    fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        damaged_count = sum(1 for conn in self.connections if conn.get('max_damage', 0.0) > self.damage_threshold)
        if damaged_count > 0:
            damage_text = f'DAMAGED SEGMENTS: {damaged_count}/{len(self.connections)}'
            self.ax.text(50, ylim - 90, damage_text, 
                        fontsize=10, va='top', color='red', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        self.fig.canvas.draw()
    
    def find_node_at_position(self, x, y):
        for fiber_idx, fiber in enumerate(self.fibers):
            for node_idx, node in enumerate(fiber):
                dist = np.sqrt((node['x'] - x)**2 + (node['y'] - y)**2)
                if dist < self.node_radius * 2:
                    return fiber_idx, node_idx
        return None, None
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        fiber_idx, node_idx = self.find_node_at_position(event.xdata, event.ydata)
        if fiber_idx is not None:
            self.dragging_info = {'fiber_idx': fiber_idx, 'node_idx': node_idx}
            node = self.fibers[fiber_idx][node_idx]
            node['fixed'] = True
            node['vx'] = 0
            node['vy'] = 0
    
    def on_release(self, event):
        if self.dragging_info is not None:
            fiber_idx = self.dragging_info['fiber_idx']
            node_idx = self.dragging_info['node_idx']
            self.fibers[fiber_idx][node_idx]['fixed'] = False
            self.dragging_node = None
            
            # IMPORTANT: Check for damage BEFORE running settling physics
            # This way, if damage occurs, the settling uses the updated rest lengths
            damage_occurred = self.check_for_damage()
            
            # Now let physics settle with the potentially updated parameters
            if self.physics_enabled:
                for _ in range(100):
                    self.simulate_physics()
            
            if damage_occurred:
                print("⚠️ Grid damaged! Entire fibers deformed when any segment turns red.")
            
            self.dragging_info = None
            self.draw()
    
    def on_motion(self, event):
        if event.inaxes != self.ax or self.dragging_info is None:
            return
        
        fiber_idx = self.dragging_info['fiber_idx']
        node_idx = self.dragging_info['node_idx']
        node = self.fibers[fiber_idx][node_idx]
        
        node['x'] = event.xdata
        node['y'] = event.ydata
        
        if self.physics_enabled:
            for _ in range(self.iterations):
                self.simulate_physics()
        
        self.draw()
    
    def run(self):
        self.draw()
        plt.show()


if __name__ == "__main__":
    model = CollagenFiberGridModel()
    print("Instructions:")
    print("- Drag any node to see realistic three-phase collagen mechanics")
    print("- Grid shows horizontal and vertical fibers intersecting")
    print("- BLUE/CYAN: Toe region (elastic, uncrimping)")
    print("- YELLOW: Linear region (elastic, stiffening)")
    print("- ORANGE: Stiff region (elastic, high resistance)")
    print("- BRIGHT RED: DAMAGE - permanently deforms ENTIRE FIBER!")
    print("- When ANY segment turns red, the WHOLE fiber it belongs to gets damaged")
    print("- All segments and nodes in that fiber update to the deformed state")
    print("- Cross-connections and boundaries are not affected by fiber damage")
    print("- 'Crimp Memory' slider: Controls recovery strength")
    print("- Click 'Reset' to restore to original undamaged state")
    model.run()