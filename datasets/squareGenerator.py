import random
import json
import math
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not needed, but recommended to visualize the datasets. pip install matplotlib")
    plt = False

NOISE_LEVEL = 0.01
GRID_SIZE = 32


def pointsToGrid(points):
    """
    Converts a list of continuous (x, y) coordinates into a 32x32 grid.
    Maps coordinates from the range [-1.5, 1.5] down to pixel indices [0, 31].
    Background initialized to -0.5, strokes assigned to 0.5.
    """
    grid = [[-0.5 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    minVal, maxVal = -1.5, 1.5
    
    for x, y in points:
        col = int(((x - minVal) / (maxVal - minVal)) * (GRID_SIZE - 1))
        row = int(((y - minVal) / (maxVal - minVal)) * (GRID_SIZE - 1))
        
        col = max(0, min(GRID_SIZE - 1, col))
        row = max(0, min(GRID_SIZE - 1, row))
        
        grid[(GRID_SIZE - 1) - row][col] = 0.5
        
    return grid


def addNoise(x, y):
    """Helper to apply a jitter offset to coordinates."""
    return [
        x + random.uniform(-NOISE_LEVEL, NOISE_LEVEL),
        y + random.uniform(-NOISE_LEVEL, NOISE_LEVEL)
    ]


def generateSquare():
    cx = random.uniform(-0.6, 0.6)
    cy = random.uniform(-0.6, 0.6)
    size = random.uniform(0.4, 1.2)
    half = size / 2

    points = []
    steps = int(size * 50)
    
    for i in range(steps):
        t = (i / steps) * size - half
        points.append(addNoise(cx - half, cy + t)) #< Left
        points.append(addNoise(cx + half, cy + t)) #< Right
        points.append(addNoise(cx + t, cy - half)) #< Bottom
        points.append(addNoise(cx + t, cy + half)) #< Top
        
    return pointsToGrid(points)


def generateCircle():
    cx = random.uniform(-0.6, 0.6)
    cy = random.uniform(-0.6, 0.6)
    radius = random.uniform(0.2, 0.6)
    
    points = []
    steps = int(2 * math.pi * radius * 50)
    
    for i in range(steps):
        angle = (i / steps) * 2 * math.pi
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append(addNoise(x, y))
        
    return pointsToGrid(points)


def generateTriangle():
    cx = random.uniform(-0.6, 0.6)
    cy = random.uniform(-0.6, 0.6)
    r = random.uniform(0.3, 0.7)
    
    # Calculate vertices of an equilateral-ish triangle
    vertices = []
    base_angle = random.uniform(0, 2 * math.pi)
    for i in range(3):
        angle = base_angle + (i * 2 * math.pi / 3)
        vertices.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        
    points = []
    steps_per_side = 40
    for i in range(3):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 3]
        for s in range(steps_per_side):
            t = s / steps_per_side
            # Linear interpolation between vertices
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            points.append(addNoise(x, y))
            
    return pointsToGrid(points)


def generateNoiseCloud():
    """Fallback variant: pure random scatter point coordinates."""
    points = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(40)]
    return pointsToGrid(points)


def generateNonSquare():
    """Randomly switches between alternative classification shapes."""
    choice = random.choice(['circle', 'triangle', 'noise'])
    if choice == 'circle':
        return generateCircle()
    elif choice == 'triangle':
        return generateTriangle()
    else:
        return generateNoiseCloud()


def generateDataset(numSamples=10000):
    X = []
    y = []

    for _ in range(numSamples):
        if random.random() < 0.5:
            X.append(generateSquare())
            y.append(1)
        else:
            X.append(generateNonSquare())
            y.append(0)

    return {"X": X, "y": y}


def visualizeSamples(dataset):
    xData = dataset["X"]
    yData = dataset["y"]

    squareIndices = [i for i, label in enumerate(yData) if label == 1][:5]
    nonSquareIndices = [i for i, label in enumerate(yData) if label == 0][:5]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, idx in enumerate(squareIndices):
        ax = axes[0, i]
        ax.imshow(xData[idx], cmap='gray_r', origin='upper', vmin=-0.5, vmax=0.5)
        ax.set_title(f"Square #{i+1}")
        ax.set_xticks(range(0, 32, 8))  
        ax.set_yticks(range(0, 32, 8))
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    for i, idx in enumerate(nonSquareIndices):
        ax = axes[1, i]
        ax.imshow(xData[idx], cmap='gray_r', origin='upper', vmin=-0.5, vmax=0.5)
        ax.set_title(f"NonSquare #{i+1}")
        ax.set_xticks(range(0, 32, 8))
        ax.set_yticks(range(0, 32, 8))
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.show()


dataset = generateDataset(1000)

with open("squareData.json", "w") as f:
    json.dump(dataset, f, indent=4)

if plt != False:
    visualizeSamples(dataset)