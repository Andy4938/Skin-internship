import pygame
import numpy as np
from typing import List, Tuple

pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 60
NUM_POINTS = 20
SPRING_CONSTANT = 0.5
DAMPING = 0.98
GRAVITY = 0.0
POINT_RADIUS = 6
DRAG_RADIUS = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (100, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.pinned = False

    def apply_force(self, fx: float, fy: float):
        if not self.pinned:
            self.vx += fx
            self.vy += fy

    def update(self):
        if not self.pinned:
            self.vy += GRAVITY
            self.vx *= DAMPING
            self.vy *= DAMPING
            self.x += self.vx
            self.y += self.vy

    def set_position(self, x: float, y: float):
        self.x = x
        self.y = y

class CollagenFiber:
    def __init__(self, start_x: float, start_y: float):
        self.points: List[Point] = []
        self.dragging_index = -1
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        
        # Create initial straight fiber
        for i in range(NUM_POINTS):
            x = start_x + i * 30
            y = start_y
            self.points.append(Point(x, y))
    
    def handle_mouse_down(self, mouse_x: int, mouse_y: int):
        for i, point in enumerate(self.points):
            dist = np.sqrt((point.x - mouse_x)**2 + (point.y - mouse_y)**2)
            if dist < DRAG_RADIUS:
                self.dragging_index = i
                self.drag_offset_x = point.x - mouse_x
                self.drag_offset_y = point.y - mouse_y
                point.pinned = True
                break
    
    def handle_mouse_motion(self, mouse_x: int, mouse_y: int):
        if self.dragging_index >= 0:
            point = self.points[self.dragging_index]
            point.set_position(
                mouse_x + self.drag_offset_x,
                mouse_y + self.drag_offset_y
            )
    
    def handle_mouse_up(self):
        if self.dragging_index >= 0:
            self.points[self.dragging_index].pinned = False
        self.dragging_index = -1
    
    def apply_constraints(self):
        # Spring constraints between adjacent points
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dist = np.sqrt(dx**2 + dy**2) + 0.0001
            rest_length = 30
            
            force = SPRING_CONSTANT * (dist - rest_length)
            fx = (force * dx) / dist
            fy = (force * dy) / dist
            
            if not p2.pinned:
                p2.apply_force(-fx, -fy)
            if not p1.pinned:
                p1.apply_force(fx, fy)
    
    def update(self):
        self.apply_constraints()
        for point in self.points:
            point.update()
        
        # Keep points in bounds
        for point in self.points:
            point.x = max(POINT_RADIUS, min(WIDTH - POINT_RADIUS, point.x))
            point.y = max(POINT_RADIUS, min(HEIGHT - POINT_RADIUS, point.y))
    
    def draw(self, surface):
        # Draw connections
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            pygame.draw.line(surface, BLUE, (p1.x, p1.y), (p2.x, p2.y), 2)
        
        # Draw points
        for i, point in enumerate(self.points):
            color = RED if i == self.dragging_index else GREEN
            pygame.draw.circle(surface, color, (int(point.x), int(point.y)), POINT_RADIUS)

def main():
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Collagen Fiber Stretching Simulator")
    
    fiber = CollagenFiber(100, HEIGHT // 2)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                fiber.handle_mouse_down(*pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONUP:
                fiber.handle_mouse_up()
        
        fiber.handle_mouse_motion(*pygame.mouse.get_pos())
        fiber.update()
        
        screen.fill(WHITE)
        fiber.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()