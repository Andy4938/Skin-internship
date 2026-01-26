import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

class CollagenFiberModel:
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.base_stiffness = 0.1  # Very low initial stiffness (toe region)
        self.max_stiffness = 2.0   # High stiffness when fully straightened
        self.iterations = 10
        self.dragging_node = None
        self.node_radius = 15
        self.num_nodes = 20
        self.zoom_level = 2.0
        self.base_xlim = 800
        self.base_ylim = 650
        self.physics_enabled = True
        self.damping = 0.65
        self.shape_memory_strength = 0.005
        self.damage_threshold = 0.95  # Straightness above this causes permanent damage
        self.max_damage = 0.0  # Track maximum damage experienced
        
        # Set up the plot with space for sliders
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.patch.set_facecolor('white')
        plt.subplots_adjust(bottom=0.2, left=0.15)
        
        # Initialize a single fiber
        self.initialize_fiber()
        
        self.setup_plot()
        self.setup_sliders()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def initialize_fiber(self):
        """Create a single collagen fiber as a chain of nodes"""
        self.nodes = []
        self.connections = []
        
        fiber_length = 600
        
        # Create nodes along a sine wave (natural crimped state)
        amplitude = 80  # Height of the crimp wave
        frequency = 2   # Number of crimp cycles
        for i in range(self.num_nodes):
            x = 100 + (i / (self.num_nodes - 1)) * fiber_length
            y = 350 + amplitude * np.sin(frequency * 2 * np.pi * i / (self.num_nodes - 1))
            self.nodes.append({
                'id': i,
                'x': x,
                'y': y,
                'vx': 0,
                'vy': 0,
                'fixed': i == 0  # Fix the leftmost node
            })
        
        # Create connections with rest length in crimped state
        for i in range(self.num_nodes - 1):
            rest_length = self.distance(i, i + 1)
            
            # The crimped fiber can be stretched beyond its rest length
            # Maximum length is when it's pulled completely straight
            max_stretch_ratio = 1.5  # Collagen can stretch ~50% before damage
            straight_length = rest_length * max_stretch_ratio
            
            self.connections.append({
                'node1': i,
                'node2': i + 1,
                'rest_length': rest_length,  # Crimped rest length
                'straight_length': straight_length,  # Maximum stretched length
                'stiffness': self.base_stiffness
            })
        
        # Store original positions for shape memory
        self.original_positions = [(node['x'], node['y']) for node in self.nodes]
    
    def distance(self, node1_idx, node2_idx):
        """Calculate distance between two nodes"""
        n1 = self.nodes[node1_idx]
        n2 = self.nodes[node2_idx]
        return np.sqrt((n2['x'] - n1['x'])**2 + (n2['y'] - n1['y'])**2)
    
    def calculate_straightness(self, conn):
        """Calculate how straight a connection is (0 = crimped, 1 = fully straight)"""
        current_length = self.distance(conn['node1'], conn['node2'])
        rest_length = conn['rest_length']
        straight_length = conn['straight_length']
        
        # Straightness measures how much the segment has been stretched
        # 0 = at rest length (crimped)
        # 1 = at maximum stretch (straight_length)
        
        if current_length <= rest_length:
            return 0.0  # Compressed or at rest
        
        straightness = (current_length - rest_length) / (straight_length - rest_length)
        return np.clip(straightness, 0.0, 1.0)
    
    def get_max_straightness(self):
        """Calculate the maximum straightness across all connections"""
        max_straightness = 0.0
        max_conn_info = None
        
        for conn in self.connections:
            straightness = self.calculate_straightness(conn)
            if straightness > max_straightness:
                max_straightness = straightness
                max_conn_info = {
                    'straightness': straightness,
                    'current': self.distance(conn['node1'], conn['node2']),
                    'rest': conn['rest_length'],
                    'max': conn['straight_length'],
                    'node1': conn['node1'],
                    'node2': conn['node2']
                }
        
        return max_straightness, max_conn_info
    
    def get_adaptive_stiffness(self, conn):
        """Calculate stiffness based on how straight the fiber is"""
        straightness = self.calculate_straightness(conn)
        
        # Three-phase behavior:
        # Phase 1 (0.0 - 0.3): Toe region - very compliant (uncrimping)
        # Phase 2 (0.3 - 0.8): Linear region - increasing stiffness
        # Phase 3 (0.8 - 1.0): Stiff region - high resistance
        
        if straightness < 0.3:
            # Toe region: very low stiffness
            return self.base_stiffness
        elif straightness < 0.8:
            # Linear region: progressive stiffening
            progress = (straightness - 0.3) / 0.5
            return self.base_stiffness + (self.max_stiffness - self.base_stiffness) * progress * 0.5
        else:
            # Stiff region: high stiffness
            progress = (straightness - 0.8) / 0.2
            return self.base_stiffness + (self.max_stiffness - self.base_stiffness) * (0.5 + progress * 0.5)
    
    def simulate_physics(self):
        """Run physics simulation with realistic collagen behavior"""
        for _ in range(self.iterations):
            # Apply spring forces with adaptive stiffness
            for conn in self.connections:
                n1 = self.nodes[conn['node1']]
                n2 = self.nodes[conn['node2']]
                
                if n1['fixed'] and n2['fixed']:
                    continue
                
                # Calculate current distance and displacement
                dx = n2['x'] - n1['x']
                dy = n2['y'] - n1['y']
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance == 0:
                    continue
                
                # Use adaptive stiffness based on straightness
                stiffness = self.get_adaptive_stiffness(conn)
                
                # Force based on displacement from rest length
                displacement = distance - conn['rest_length']
                
                # Cap the force to prevent explosions
                max_displacement = conn['straight_length'] - conn['rest_length']
                displacement = np.clip(displacement, -max_displacement, max_displacement)
                
                force = displacement * stiffness
                
                # Calculate force components
                fx = (dx / distance) * force
                fy = (dy / distance) * force
                
                # Apply forces to velocities
                if not n1['fixed']:
                    n1['vx'] += fx
                    n1['vy'] += fy
                
                if not n2['fixed']:
                    n2['vx'] -= fx
                    n2['vy'] -= fy
            
            # Apply shape memory forces (restore crimp pattern)
            # Only apply when NOT dragging, or far from dragged node
            if self.dragging_node is None:
                for i, node in enumerate(self.nodes):
                    if not node['fixed'] and i < len(self.original_positions):
                        dx_orig = self.original_positions[i][0] - node['x']
                        dy_orig = self.original_positions[i][1] - node['y']
                        
                        # Reduce shape memory strength based on damage
                        # If damage > threshold, fiber loses ability to return to original crimp
                        if self.max_damage > self.damage_threshold:
                            damage_factor = 1.0 - ((self.max_damage - self.damage_threshold) / (1.0 - self.damage_threshold))
                            damage_factor = max(0.0, damage_factor)  # 0 = fully damaged, 1 = undamaged
                        else:
                            damage_factor = 1.0
                        
                        node['vx'] += dx_orig * self.shape_memory_strength * damage_factor
                        node['vy'] += dy_orig * self.shape_memory_strength * damage_factor
            
            # Update positions and apply damping
            for node in self.nodes:
                if not node['fixed']:
                    node['vx'] *= self.damping
                    node['vy'] *= self.damping
                    
                    # Cap velocity to prevent nodes from flying off screen
                    max_velocity = 50
                    node['vx'] = np.clip(node['vx'], -max_velocity, max_velocity)
                    node['vy'] = np.clip(node['vy'], -max_velocity, max_velocity)
                    
                    node['x'] += node['vx']
                    node['y'] += node['vy']
    
    def setup_plot(self):
        """Set up the matplotlib plot"""
        xlim = self.base_xlim * self.zoom_level
        ylim = self.base_ylim * self.zoom_level
        
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        self.ax.text(xlim/2, ylim - 30, 'Realistic Collagen Fiber - Three-Phase Mechanical Response', 
                    ha='center', va='top', fontsize=14, fontweight='bold')
    
    def setup_sliders(self):
        """Set up control sliders"""
        # Number of nodes slider
        slider_ax1 = plt.axes([0.25, 0.11, 0.5, 0.03])
        self.slider_nodes = Slider(
            ax=slider_ax1,
            label='Number of Nodes',
            valmin=5,
            valmax=50,
            valinit=self.num_nodes,
            valstep=1,
            color='steelblue'
        )
        self.slider_nodes.on_changed(self.update_num_nodes)
        
        # Zoom slider
        slider_ax2 = plt.axes([0.25, 0.06, 0.5, 0.03])
        self.slider_zoom = Slider(
            ax=slider_ax2,
            label='Zoom Out',
            valmin=1.0,
            valmax=5.0,
            valinit=self.zoom_level,
            valfmt='%.1fx',
            color='green'
        )
        self.slider_zoom.on_changed(self.update_zoom)
        
        # Damping slider
        slider_ax3 = plt.axes([0.25, 0.01, 0.5, 0.03])
        self.slider_damping = Slider(
            ax=slider_ax3,
            label='Damping',
            valmin=0.5,
            valmax=0.99,
            valinit=self.damping,
            valfmt='%.2f',
            color='orange'
        )
        self.slider_damping.on_changed(self.update_damping)
        
        # Shape memory slider (vertical)
        slider_ax4 = plt.axes([0.08, 0.25, 0.02, 0.5])
        self.slider_shape_memory = Slider(
            ax=slider_ax4,
            label='Crimp\nMemory',
            valmin=0.0,
            valmax=0.1,
            valinit=self.shape_memory_strength,
            valfmt='%.3f',
            color='teal',
            orientation='vertical'
        )
        self.slider_shape_memory.on_changed(self.update_shape_memory)
        
        # Physics toggle button
        from matplotlib.widgets import Button
        button_ax = plt.axes([0.85, 0.12, 0.12, 0.04])
        self.button_physics = Button(button_ax, 'Physics: ON', color='lightgreen')
        self.button_physics.on_clicked(self.toggle_physics)
        
        # Reset button
        button_ax_reset = plt.axes([0.85, 0.06, 0.12, 0.04])
        self.button_reset = Button(button_ax_reset, 'Reset', color='lightcoral')
        self.button_reset.on_clicked(self.reset)
    
    def update_num_nodes(self, val):
        """Update the number of nodes when slider changes"""
        self.num_nodes = int(val)
        self.dragging_node = None
        self.initialize_fiber()
        self.draw()
    
    def update_zoom(self, val):
        """Update the zoom level when slider changes"""
        self.zoom_level = val
        self.draw()
    
    def update_damping(self, val):
        """Update the damping factor when slider changes"""
        self.damping = val
    
    def update_shape_memory(self, val):
        """Update the shape memory strength when slider changes"""
        self.shape_memory_strength = val
    
    def toggle_physics(self, event):
        """Toggle physics simulation on/off"""
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
        """Reset the fiber to its initial state"""
        self.dragging_node = None
        self.max_damage = 0.0  # Reset damage tracking
        self.initialize_fiber()
        self.draw()
        print("Fiber reset to original undamaged state")
    
    def draw(self):
        """Draw the fiber with color coding"""
        self.ax.clear()
        self.setup_plot()
        
        # Draw the fiber with color coding based on straightness
        for conn in self.connections:
            n1 = self.nodes[conn['node1']]
            n2 = self.nodes[conn['node2']]
            
            # Calculate straightness (0 = crimped, 1 = straight)
            straightness = self.calculate_straightness(conn)
            
            # Color coding based on straightness and damage threshold:
            # Blue/Cyan (0.0-0.3): Toe region - uncrimping, always elastic
            # Yellow (0.3-0.8): Linear region - stiffening, elastic
            # Orange (0.8-0.95): Stiff region - high resistance, still elastic
            # Bright Red (0.95-1.0): DAMAGE ZONE - permanent deformation
            
            if straightness < 0.3:
                # Toe region: blue to cyan (safe)
                intensity = straightness / 0.3
                color = (0.2, 0.5 + 0.5*intensity, 1.0)
            elif straightness < 0.8:
                # Linear region: cyan to yellow (safe)
                intensity = (straightness - 0.3) / 0.5
                r = 0.2 + 0.8 * intensity
                g = 1.0
                b = 1.0 - intensity
                color = (r, g, b)
            elif straightness < self.damage_threshold:
                # Stiff region: yellow to orange (still safe!)
                intensity = (straightness - 0.8) / (self.damage_threshold - 0.8)
                color = (1.0, 1.0 - 0.5*intensity, 0.0)  # Yellow to orange
            else:
                # DAMAGE ZONE: bright red warning
                intensity = min((straightness - self.damage_threshold) / (1.0 - self.damage_threshold), 1.0)
                color = (1.0, 0.0, 0.0)  # Pure red for damage
            
            self.ax.plot([n1['x'], n2['x']], [n1['y'], n2['y']], 
                        color=color, linewidth=4, alpha=0.8, solid_capstyle='round')
        
        # Draw nodes
        for node in self.nodes:
            if node['fixed']:
                color = 'darkred'
                size = self.node_radius * 1.2
            else:
                color = 'steelblue'
                size = self.node_radius
            
            circle = Circle((node['x'], node['y']), size, 
                          color=color, alpha=0.8, zorder=10, edgecolor='white', linewidth=2)
            self.ax.add_patch(circle)
        
        # Add legend
        ylim = self.base_ylim * self.zoom_level
        legend_text = 'Blue/Cyan: Elastic | Yellow: Stiffening | Orange: Very Stiff | RED: DAMAGE!'
        self.ax.text(50, ylim - 40, legend_text, 
                    fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Show damage status
        if self.max_damage > self.damage_threshold:
            damage_pct = int(((self.max_damage - self.damage_threshold) / (1.0 - self.damage_threshold)) * 100)
            damage_text = f'DAMAGED: {damage_pct}% (Permanent deformation)'
            self.ax.text(50, ylim - 70, damage_text, 
                        fontsize=10, va='top', color='red', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        self.fig.canvas.draw()
    
    def find_node_at_position(self, x, y):
        """Find which node (if any) is at the given position"""
        for i, node in enumerate(self.nodes):
            dist = np.sqrt((node['x'] - x)**2 + (node['y'] - y)**2)
            if dist < self.node_radius * 1.5:
                return i
        return None
    
    def on_press(self, event):
        """Handle mouse press event"""
        if event.inaxes != self.ax:
            return
        
        node_idx = self.find_node_at_position(event.xdata, event.ydata)
        if node_idx is not None and node_idx != 0:
            self.dragging_node = node_idx
            self.nodes[node_idx]['fixed'] = True
            self.nodes[node_idx]['vx'] = 0
            self.nodes[node_idx]['vy'] = 0
    
    def on_release(self, event):
        """Handle mouse release event"""
        if self.dragging_node is not None:
            self.nodes[self.dragging_node]['fixed'] = False
            self.dragging_node = None
            
            # Run extra physics iterations to let the fiber settle
            if self.physics_enabled:
                for _ in range(100):
                    self.simulate_physics()
            
            # NOW check for damage after the fiber has settled
            max_straightness, max_conn_info = self.get_max_straightness()
            
            # Update max damage if we've exceeded previous maximum
            if max_straightness > self.max_damage:
                self.max_damage = max_straightness
            
            # Update the "memory" positions to current positions if damaged
            # This creates plastic deformation - fiber remembers its new damaged state
            if self.max_damage > self.damage_threshold:
                self.original_positions = [(node['x'], node['y']) for node in self.nodes]
                print(f"⚠️ FIBER DAMAGED! Max straightness reached: {self.max_damage:.3f} (threshold: {self.damage_threshold})")
                if max_conn_info:
                    print(f"   Damaged segment: nodes {max_conn_info['node1']}-{max_conn_info['node2']}")
                    print(f"   Current length: {max_conn_info['current']:.2f}, Rest: {max_conn_info['rest']:.2f}, Max: {max_conn_info['max']:.2f}")
                damage_pct = int(((self.max_damage - self.damage_threshold) / (1.0 - self.damage_threshold)) * 100)
                print(f"   New rest state set. Damage: {damage_pct}%")
            else:
                print(f"✓ No damage. Max straightness: {max_straightness:.3f} (threshold: {self.damage_threshold})")
            
            self.draw()
    
    def on_motion(self, event):
        """Handle mouse motion event"""
        if event.inaxes != self.ax or self.dragging_node is None:
            return
        
        self.nodes[self.dragging_node]['x'] = event.xdata
        self.nodes[self.dragging_node]['y'] = event.ydata
        
        if self.physics_enabled:
            # Run more iterations to let nodes settle during dragging
            for _ in range(30):  # Increased from 10 to allow better settling
                self.simulate_physics()
            
            # Check for damage after nodes have had time to propagate the stretch
            max_straightness, max_conn_info = self.get_max_straightness()
            
            # Update max damage if we've exceeded previous maximum
            if max_straightness > self.max_damage:
                self.max_damage = max_straightness
                
                # If we've crossed into damage territory, update memory positions
                if self.max_damage > self.damage_threshold:
                    self.original_positions = [(node['x'], node['y']) for node in self.nodes]
                    print(f"⚠️ FIBER DAMAGED during drag! Straightness: {self.max_damage:.3f}")
                    if max_conn_info:
                        print(f"   Damaged segment: nodes {max_conn_info['node1']}-{max_conn_info['node2']}")
        
        self.draw()
    
    def run(self):
        """Start the interactive visualization"""
        self.draw()
        plt.show()


# Run the model
if __name__ == "__main__":
    model = CollagenFiberModel()
    print("Instructions:")
    print("- Drag nodes to see realistic three-phase collagen mechanics")
    print("- BLUE/CYAN: Toe region (elastic, uncrimping)")
    print("- YELLOW: Linear region (elastic, stiffening)")
    print("- ORANGE: Stiff region (elastic, high resistance)")
    print("- BRIGHT RED: DAMAGE ZONE - causes permanent deformation!")
    print("- 'Crimp Memory' slider: Controls how strongly fiber returns to wavy shape")
    print("- Watch how the fiber gets stiffer as you straighten it!")
    print("- Release to see it recover its crimped pattern")
    print("- Only segments that turn BRIGHT RED cause permanent damage")
    print("- After damage, the fiber remembers its new straightened state")
    print("- Click 'Reset' to restore to original undamaged state")
    print("\nFIXED: Damage now only checked after release when fiber has settled!")
    model.run()