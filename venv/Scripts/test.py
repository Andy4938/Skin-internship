import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

class CollagenFiberModel:
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.stiffness = 0.5
        self.iterations = 50
        self.dragging_node = None
        self.node_radius = 15  # pixels for visibility
        self.num_nodes = 20
        self.zoom_level = 1.0
        self.base_xlim = 800
        self.base_ylim = 650
        self.physics_enabled = True
        
        # Set up the plot with space for sliders
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.patch.set_facecolor('white')
        plt.subplots_adjust(bottom=0.2)
        
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
        
        # Create nodes along a line
        for i in range(self.num_nodes):
            x = 100 + (i / (self.num_nodes - 1)) * fiber_length
            y = 350
            self.nodes.append({
                'id': i,
                'x': x,
                'y': y,
                'fixed': i == 0  # Fix the leftmost node
            })
        
        # Create connections between consecutive nodes
        for i in range(self.num_nodes - 1):
            rest_length = self.distance(i, i + 1)
            # Higher stiffness for the first connection to provide counter force
            stiffness = self.stiffness * 2 if i == 0 else self.stiffness
            self.connections.append({
                'node1': i,
                'node2': i + 1,
                'rest_length': rest_length,
                'stiffness': stiffness
            })
    
    def distance(self, node1_idx, node2_idx):
        """Calculate distance between two nodes"""
        n1 = self.nodes[node1_idx]
        n2 = self.nodes[node2_idx]
        return np.sqrt((n2['x'] - n1['x'])**2 + (n2['y'] - n1['y'])**2)
    
    def simulate_physics(self):
        """Run physics simulation to relax the fiber"""
        for _ in range(self.iterations):
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
                
                # Apply forces to nodes (if not fixed)
                if not n1['fixed']:
                    n1['x'] += fx
                    n1['y'] += fy
                
                if not n2['fixed']:
                    n2['x'] -= fx
                    n2['y'] -= fy
    
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
        self.ax.text(xlim/2, ylim - 30, 'Single Collagen Fiber - Drag nodes to stretch', 
                    ha='center', va='top', fontsize=14, fontweight='bold')
    
    def setup_sliders(self):
        """Set up the sliders for changing number of nodes and zoom"""
        # Number of nodes slider
        slider_ax1 = plt.axes([0.25, 0.08, 0.5, 0.03])
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
        slider_ax2 = plt.axes([0.25, 0.03, 0.5, 0.03])
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
        
        # Physics toggle button
        from matplotlib.widgets import Button
        button_ax = plt.axes([0.8, 0.08, 0.12, 0.04])
        self.button_physics = Button(button_ax, 'Physics: ON', color='lightgreen')
        self.button_physics.on_clicked(self.toggle_physics)
    
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
    
    def on_release(self, event):
        """Handle mouse release event"""
        if self.dragging_node is not None:
            self.nodes[self.dragging_node]['fixed'] = False
            self.dragging_node = None
    
    def on_motion(self, event):
        """Handle mouse motion event"""
        if event.inaxes != self.ax or self.dragging_node is None:
            return
        
        # Update dragged node position
        self.nodes[self.dragging_node]['x'] = event.xdata
        self.nodes[self.dragging_node]['y'] = event.ydata
        
        # Run physics simulation only if enabled
        if self.physics_enabled:
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
    print("- Blue segments = compressed, Red segments = stretched")
    print("- The fiber will relax based on spring physics")
    print("- Click 'Physics: ON/OFF' button to toggle physics simulation")
    print("- When physics is OFF, nodes move freely without affecting neighbors")
    print("- Close the window to exit")
    model.run()