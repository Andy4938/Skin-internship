import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

class CollagenFiberModel:
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.stiffness = 0.5
        self.iterations = 10  # Reduced for smoother real-time interaction
        self.dragging_node = None
        self.node_radius = 15  # pixels for visibility
        self.num_nodes = 20
        self.zoom_level = 1.0
        self.base_xlim = 800
        self.base_ylim = 650
        self.physics_enabled = True
        self.damping = 0.85  # Damping factor (0-1, lower = more damping)
        self.bending_stiffness = 0.02  # Resistance to changes in angle between segments
        
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
        
        # Create nodes along a sine wave (this is the natural relaxed state)
        amplitude = 80  # Height of the wave
        frequency = 2   # Number of wave cycles
        for i in range(self.num_nodes):
            x = 100 + (i / (self.num_nodes - 1)) * fiber_length
            # Add sine wave variation - this is the NATURAL shape
            y = 350 + amplitude * np.sin(frequency * 2 * np.pi * i / (self.num_nodes - 1))
            self.nodes.append({
                'id': i,
                'x': x,
                'y': y,
                'vx': 0,  # velocity x component
                'vy': 0,  # velocity y component
                'fixed': i == 0  # Fix the leftmost node
            })
        
        # Create connections between consecutive nodes
        # IMPORTANT: Rest lengths are calculated AFTER positioning nodes in wave pattern
        # This makes the wavy shape the natural, relaxed configuration
        for i in range(self.num_nodes - 1):
            rest_length = self.distance(i, i + 1)  # This captures the curved rest state
            # Higher stiffness for the first connection to provide counter force
            stiffness = self.stiffness * 2 if i == 0 else self.stiffness
            self.connections.append({
                'node1': i,
                'node2': i + 1,
                'rest_length': rest_length,  # The wavy distance is the natural rest length
                'stiffness': stiffness
            })
        
        # NEW: Store rest angles for bending resistance
        # Calculate the natural angle between consecutive segments
        self.rest_angles = []
        for i in range(self.num_nodes - 2):
            angle = self.calculate_angle(i, i + 1, i + 2)
            self.rest_angles.append(angle)
    
    def distance(self, node1_idx, node2_idx):
        """Calculate distance between two nodes"""
        n1 = self.nodes[node1_idx]
        n2 = self.nodes[node2_idx]
        return np.sqrt((n2['x'] - n1['x'])**2 + (n2['y'] - n1['y'])**2)
    
    def calculate_angle(self, node1_idx, node2_idx, node3_idx):
        """Calculate the angle at node2 formed by three consecutive nodes"""
        n1 = self.nodes[node1_idx]
        n2 = self.nodes[node2_idx]
        n3 = self.nodes[node3_idx]
        
        # Vectors from node2 to node1 and node2 to node3
        v1x = n1['x'] - n2['x']
        v1y = n1['y'] - n2['y']
        v2x = n3['x'] - n2['x']
        v2y = n3['y'] - n2['y']
        
        # Calculate angle using atan2 for proper quadrant handling
        angle1 = np.arctan2(v1y, v1x)
        angle2 = np.arctan2(v2y, v2x)
        
        # Calculate angular difference
        angle_diff = angle2 - angle1
        
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        return angle_diff
    
    def simulate_physics(self):
        """Run physics simulation with damping and bending resistance"""
        for _ in range(self.iterations):
            # Apply spring forces (axial/stretching)
            for conn in self.connections:
                n1 = self.nodes[conn['node1']]
                n2 = self.nodes[conn['node2']]
                
                # Skip if both nodes are fixed
                if n1['fixed'] and n2['fixed']:
                    continue
                
                # Calculate current distance and displacement
                dx = n2['x'] - n1['x']
                dy = n2['y'] - n1['y']
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance == 0:
                    continue
                
                # Calculate force based on displacement from rest length
                stiffness = conn.get('stiffness', self.stiffness)
                displacement = distance - conn['rest_length']
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
            
            # NEW: Apply bending resistance forces
            for i in range(len(self.rest_angles)):
                n1 = self.nodes[i]
                n2 = self.nodes[i + 1]
                n3 = self.nodes[i + 2]
                
                # Skip if all nodes are fixed
                if n1['fixed'] and n2['fixed'] and n3['fixed']:
                    continue
                
                # NEW: Calculate distance-based attenuation if a node is being dragged
                distance_factor = 1.0
                if self.dragging_node is not None:
                    # Calculate the minimum distance from this triplet to the dragged node
                    min_dist = min(
                        abs(i - self.dragging_node),
                        abs(i + 1 - self.dragging_node),
                        abs(i + 2 - self.dragging_node)
                    )
                    # Exponential decay: forces decay as e^(-distance/2)
                    # At distance 0: factor = 1.0
                    # At distance 2: factor ≈ 0.37
                    # At distance 4: factor ≈ 0.14
                    # At distance 6: factor ≈ 0.05
                    distance_factor = np.exp(-min_dist / 2.0)
                
                # Calculate current angle
                current_angle = self.calculate_angle(i, i + 1, i + 2)
                rest_angle = self.rest_angles[i]
                
                # Angular displacement (how much the angle has changed)
                angle_diff = current_angle - rest_angle
                
                # Get segment lengths
                d1 = self.distance(i, i + 1)
                d2 = self.distance(i + 1, i + 2)
                
                if d1 == 0 or d2 == 0:
                    continue
                
                # Torque proportional to angular displacement
                # Scale torque by the segment lengths to make it comparable to spring forces
                # Apply distance-based attenuation
                avg_length = (d1 + d2) / 2
                torque = angle_diff * self.bending_stiffness * avg_length * distance_factor
                
                # Convert torque to forces on the three nodes
                # Calculate perpendicular forces to restore the rest angle
                # Distribute force based on segment lengths (longer segments get proportionally less force)
                force_mag1 = torque / d1
                force_mag2 = torque / d2
                
                # Get direction vectors
                v1x = (n1['x'] - n2['x']) / d1
                v1y = (n1['y'] - n2['y']) / d1
                v2x = (n3['x'] - n2['x']) / d2
                v2y = (n3['y'] - n2['y']) / d2
                
                # Perpendicular forces (rotate by 90 degrees)
                # Apply forces perpendicular to each segment
                f1x = -v1y * force_mag1
                f1y = v1x * force_mag1
                f3x = v2y * force_mag2
                f3y = -v2x * force_mag2
                
                # Apply bending forces to velocities
                if not n1['fixed']:
                    n1['vx'] += f1x
                    n1['vy'] += f1y
                
                if not n2['fixed']:
                    # Middle node gets forces from both sides
                    n2['vx'] -= (f1x + f3x)
                    n2['vy'] -= (f1y + f3y)
                
                if not n3['fixed']:
                    n3['vx'] += f3x
                    n3['vy'] += f3y
            
            # Update positions based on velocities and apply damping
            for node in self.nodes:
                if not node['fixed']:
                    # Apply damping to velocities
                    node['vx'] *= self.damping
                    node['vy'] *= self.damping
                    
                    # Update positions
                    node['x'] += node['vx']
                    node['y'] += node['vy']
    
    def setup_plot(self):
        """Set up the matplotlib plot with hidden axes"""
        xlim = self.base_xlim * self.zoom_level
        ylim = self.base_ylim * self.zoom_level
        
        self.ax.set_xlim(0, xlim)
        self.ax.set_ylim(0, ylim)
        self.ax.set_aspect('equal')
        
        # Hide all axes, ticks, and labels
        self.ax.axis('off')
        
        # Add title (position scales with zoom)
        self.ax.text(xlim/2, ylim - 30, 'Realistic Collagen Fiber - Wavy shape is natural rest state', 
                    ha='center', va='top', fontsize=14, fontweight='bold')
    
    def setup_sliders(self):
        """Set up the sliders for changing number of nodes, zoom, damping, and bending"""
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
        
        # NEW: Bending stiffness slider (vertical)
        slider_ax4 = plt.axes([0.08, 0.25, 0.02, 0.5])
        self.slider_bending = Slider(
            ax=slider_ax4,
            label='Bending\nStiffness',
            valmin=0.0,
            valmax=0.1,
            valinit=self.bending_stiffness,
            valfmt='%.3f',
            color='purple',
            orientation='vertical'
        )
        self.slider_bending.on_changed(self.update_bending_stiffness)
        
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
        self.dragging_node = None  # Stop any dragging
        self.initialize_fiber()
        self.draw()
    
    def update_zoom(self, val):
        """Update the zoom level when slider changes"""
        self.zoom_level = val
        self.draw()
    
    def update_damping(self, val):
        """Update the damping factor when slider changes"""
        self.damping = val
    
    def update_bending_stiffness(self, val):
        """Update the bending stiffness when slider changes"""
        self.bending_stiffness = val
    
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
        self.initialize_fiber()
        self.draw()
    
    def draw(self):
        """Draw the fiber"""
        self.ax.clear()
        self.setup_plot()
        
        # Draw the fiber as a continuous line with color coding
        for conn in self.connections:
            n1 = self.nodes[conn['node1']]
            n2 = self.nodes[conn['node2']]
            
            # Calculate strain for color coding
            current_length = np.sqrt((n2['x'] - n1['x'])**2 + (n2['y'] - n1['y'])**2)
            strain = (current_length - conn['rest_length']) / conn['rest_length']
            
            # Color: blue (compressed) -> gray (rest) -> red (stretched)
            if strain > 0.01:
                intensity = min(strain * 10, 1.0)
                color = (1.0, 1.0 - intensity, 1.0 - intensity)  # White to red
            elif strain < -0.01:
                intensity = min(abs(strain) * 10, 1.0)
                color = (1.0 - intensity, 1.0 - intensity, 1.0)  # White to blue
            else:
                color = (0.6, 0.6, 0.6)  # Gray for neutral
            
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
        
        # Add legend in corner (position scales with zoom)
        ylim = self.base_ylim * self.zoom_level
        self.ax.text(50, ylim - 40, 'Blue: Compressed | Gray: Neutral | Red: Stretched', 
                    fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
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
        if node_idx is not None and node_idx != 0:  # Don't allow dragging the leftmost fixed node
            self.dragging_node = node_idx
            self.nodes[node_idx]['fixed'] = True
            # Reset velocities when grabbing a node
            self.nodes[node_idx]['vx'] = 0
            self.nodes[node_idx]['vy'] = 0
    
    def on_release(self, event):
        """Handle mouse release event"""
        if self.dragging_node is not None:
            self.nodes[self.dragging_node]['fixed'] = False
            self.dragging_node = None
            # Run extra physics iterations to let the fiber settle back
            if self.physics_enabled:
                for _ in range(100):  # Extra relaxation iterations
                    self.simulate_physics()
                self.draw()
    
    def on_motion(self, event):
        """Handle mouse motion event"""
        if event.inaxes != self.ax or self.dragging_node is None:
            return
        
        # Update dragged node position
        self.nodes[self.dragging_node]['x'] = event.xdata
        self.nodes[self.dragging_node]['y'] = event.ydata
        
        # Run physics simulation only if enabled
        if self.physics_enabled:
            # Run multiple iterations to propagate forces through the fiber
            for _ in range(self.iterations):
                self.simulate_physics()
        
        # Redraw
        self.draw()
    
    def run(self):
        """Start the interactive visualization"""
        self.draw()
        plt.show()


# Run the model
if __name__ == "__main__":
    model = CollagenFiberModel()
    print("Instructions:")
    print("- Click and drag any node to stretch the collagen fiber")
    print("- The WAVY shape is the natural relaxed state (like real collagen crimp)")
    print("- Straightening the fiber creates tension - it will spring back when released")
    print("- Blue segments = compressed, Red segments = stretched")
    print("- Adjust 'Damping' slider (higher = less friction, more oscillation)")
    print("- Adjust 'Bending Stiffness' slider (higher = more resistance to bending)")
    print("- Click 'Physics: ON/OFF' button to toggle physics simulation")
    print("- Click 'Reset' button to restore the fiber to its initial state")
    print("- Close the window to exit")
    model.run()