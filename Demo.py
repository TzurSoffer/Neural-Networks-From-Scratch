import tkinter as tk
import json

# Import your custom framework modules
try:
    import mathlib as Mathlib                             #< c++ version (much faster). Must be compiled first
    from NeuralNetwork_CPP import Layer, ActivationType   #< c++ version (much faster). Must be compiled first
except ImportError:
    print("C++ modules not compiled, falling back to python.")
    import PYTHON_Network.Mathlib as Mathlib
    from PYTHON_Network.Layer import Layer
    from PYTHON_Network.ActivationTypes import ActivationType

import PYTHON_Network.Activation as Activation

try:
    with open("trainedModel.json", "r") as f:
        savedModel = json.load(f)
    print("Successfully loaded 'trainedModel.json'!")
except FileNotFoundError:
    print("Error: 'trainedModel.json' not found. Please train and save your model first.")
    exit()

layer1 = Layer(32*32, 32, ActivationType.LEAKY_RELU)
layer2 = Layer(32, 2, ActivationType.PASS)

layer1.setWeights(savedModel["layer1"]["weights"])
layer1.setBiases(savedModel["layer1"]["biases"])
layer2.setWeights(savedModel["layer2"]["weights"])
layer2.setBiases(savedModel["layer2"]["biases"])

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Square Classifier Demo")

        self.grid_size = 32
        self.pixel_size = 12  # Upscale for easier mouse drawing
        self.canvas_dim = self.grid_size * self.pixel_size
        
        # -0.5 is background, 0.5 is drawing stroke   (best for training)
        self.grid = [[-0.5 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        self.canvas = tk.Canvas(root, width=self.canvas_dim, height=self.canvas_dim, bg="white")
        self.canvas.pack()
        
        self.clear_btn = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(pady=5)
        
        self.label = tk.Label(root, text="Draw a shape!", font=("Arial", 14))
        self.label.pack(pady=5)
        
        # Mouse Bindings
        self.canvas.bind("<B1-Motion>", self.paint)        #< Holding left-click draws
        self.canvas.bind("<ButtonRelease-1>", self.infer)  #< Releasing left-click triggers the model

    def paint(self, event):
        # Map screen coords to our 32x32 grid indices
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size

        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            self.grid[row][col] = 0.5
            
            x1 = col * self.pixel_size
            y1 = row * self.pixel_size
            x2 = x1 + self.pixel_size
            y2 = y1 + self.pixel_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

    def infer(self, event):
        # Convert the 2D grid to 1D using Hilbert Curve
        flattened_X = Mathlib.hilbertFlatten(self.grid)

        out1 = layer1.forward(flattened_X)
        out2 = layer2.forward(out1)
        probabilities = Activation.Softmax.forward(out2)
        prob_non_square = probabilities[0]
        prob_square = probabilities[1]

        if prob_square > prob_non_square:
            result_str = f"SQUARE! (Confidence: {prob_square*100:.1f}%)"
        else:
            result_str = f"NOT A SQUARE! (Confidence: {prob_non_square*100:.1f}%)"

        print(f"[Model Inference] {result_str}")
        self.label.config(text=result_str)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid = [[-0.5 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.label.config(text="Draw a shape!")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()