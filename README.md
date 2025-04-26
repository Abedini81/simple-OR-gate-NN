# simple-or-gate-neural-network-from-scratch

This project implements a **basic neural network from scratch in raw Python** without using machine learning libraries.  
The neural network is trained on an **OR gate dataset**. During training, the loss values are recorded and saved into a CSV file, which is later visualized using a Jupyter Notebook.

## Project Structure
- `touch_gates.py` — Main Python script implementing and training the neural network.
- `loss_values_gates.csv` — Output CSV file containing loss values for each epoch.
- `graphs.ipynb` — Jupyter Notebook used for visualizing the loss values over time.

## How It Works
1. The neural network is initialized with random weights and bias.
2. It is trained using numerical gradient descent to minimize the mean squared error loss.
3. After training, the model's performance is tested on OR gate inputs.
4. The loss values during training are saved into a CSV file.
5. A Jupyter Notebook is used to plot the **Loss vs. Epochs** graph.

## Example Output

initial loss: [0.13294505]
W1 = [4.53749557], W2 = [4.53528661], bias = [-2.00368412], Loss = [0.00623779]
0 | 0 = [0.11881666]
0 | 1 = [0.92632779]
1 | 0 = [0.9264784]
1 | 1 = [0.99914972]

## Graph

> **[output.png]**

## Conclusion

The loss graph demonstrates that the model successfully minimized the error over training epochs.  
The training started with a relatively high loss (~0.149) and steadily decreased to a very low value (~0.006), indicating that the neural network effectively learned the OR gate logic.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Jupyter Notebook (for visualization)

## How to Run
1. Run `raw_nn.py` to train the model and generate `loss_values_gates.csv`.
2. Open `graphs.ipynb` in Jupyter Notebook to plot and visualize the loss graph.
