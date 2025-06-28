"""
Training, Evaluation, and Integration of Neural Networks for Solving Differential Equations.

This module performs:
1. Training neural networks to approximate solutions of ODEs and systems of ODEs.
2. Evaluating performance by comparing predictions to numerical solutions (Runge-Kutta).
3. Handling advanced training techniques such as adaptive sampling and loss visualization.
"""

import json
import os.path
import random
from datetime import datetime
from itertools import combinations #not in novo codigo.
from typing import List, NamedTuple, Optional, Dict, Callable, Tuple, Sequence
import numpy as np
import torch as th
import matplotlib as mpl
from matplotlib import pyplot as plt
from torchinfo import summary
from tqdm import tqdm
from scipy.stats import skewnorm
import copy

from equations import SecondOrderEquations, compute_derivative, SystemEquations
from modules import EquationsModel

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

plt.rcParams['axes.labelsize']  = 14

th.set_default_dtype(th.float64) # Set PyTorch's default tensor type to float64 for higher precision

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def assert_directory_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def set_all_seeds(seed: int):
    """Sets a fixed random seed for all possible sources of randomness to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


class Configuration(NamedTuple):
    """
    A configuration container for neural network settings.

    Attributes:
        seed (int): Random seed for reproducibility.
        features (List[int]): List defining the number of neurons per hidden layer.
        activation_function (Callable): Activation function used in the neural network.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        steps (int): Number of discrete time steps used in training.
    """
    seed: int
    features: List[int]
    activation_function: Callable[[th.Tensor], th.Tensor]
    learning_rate: float
    epochs: int
    steps: int

# ------------------------------------------------------------------
# choose the smoother you like
SMOOTH = "ema"      # "ma" | "savgol" | "ema"
WINDOW = 51       # odd number (used by all 3)
EMA_ALPHA = 2 / (WINDOW + 1)   # only for "ema"
# ------------------------------------------------------------------

def smooth_curve(arr: np.ndarray, trim_frac: float = 0.1) -> np.ndarray:
    """
    Smooth only the central (1 - 2*trim_frac) fraction of arr.
    Returns the smoothed slice; its length is len(arr) - 2*trim.
    
    Args:
      arr       : 1D array of values (e.g. losses)
      trim_frac : fraction to cut off each end (default 0.05 for 5%)
    """
    n = len(arr)
    trim = int(n * trim_frac)
    if trim * 2 >= n:
        # nothing to smooth—array too short
        return arr.copy()
    
    # central 90%
    core = arr[0 : n - trim]
    
    # apply your chosen smoother to `core`
    if SMOOTH == "ma":
        kernel = np.ones(WINDOW) / WINDOW
        sm = np.convolve(core, kernel, mode="valid")
    elif SMOOTH == "savgol":
        sm = savgol_filter(core, WINDOW, polyorder=3)
    elif SMOOTH == "ema":
        sm = np.empty_like(core, dtype=float)
        sm[0] = core[0]
        for i in range(1, len(core)):
            sm[i] = EMA_ALPHA * core[i] + (1 - EMA_ALPHA) * sm[i - 1]
    else:
        sm = core.copy()
    
    return sm

def smooth_curve2(arr: np.ndarray) -> np.ndarray:
    if SMOOTH == "ma":                       # simple moving average
        kernel = np.ones(WINDOW) / WINDOW
        return np.convolve(arr, kernel, mode="valid")
    elif SMOOTH == "savgol":                 # Savitzky‑Golay
        return savgol_filter(arr, WINDOW, polyorder=3)
    elif SMOOTH == "ema":                    # exponential moving average
        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = EMA_ALPHA * arr[i] + (1-EMA_ALPHA) * out[i-1]
        return out
    else:
        return arr

def calculate_y_limits_log(
    losses: Sequence[float],
    lower_pct: float = 2.0,     # ignora 2 % mais baixos
    upper_pct: float = 99.9,    # ignora 2 % mais altos
    headroom: float = 1.15      # margem extra no topo (15 %)
) -> Tuple[float, float]:
    """Escolhe limites y para gráfico log, cortando cauda inferior/superior."""
    arr = np.asarray(losses, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]      # segura‑log

    if arr.size == 0:
        return 1e-12, 1.0

    lo = np.percentile(arr, lower_pct)
    hi = np.percentile(arr, upper_pct)

    bottom = lo
    top    = hi

    return bottom, top

def calculate_y_limits(losses: List[float], z_threshold: float = 2.0,*, non_negative: bool = True, padding: float = 5e-4) -> Tuple[float, float]:
    """
    Ajusta os limites do eixo‑y filtrando outliers.
    
    Args:
        losses (List[float])  : histórico de loss.
        z_threshold (float)   : z‑score máximo admitido.
        non_negative (bool)   : se True força bottom ≥ 0.
        padding (float)       : margem extra acima/abaixo dos dados.
    """
    if len(losses) == 0:
        return 0.0, 1.0            # fallback seguro
    
    arr = np.asarray(losses)
    mean, std = arr.mean(), arr.std()

    if std < 1e-8:                 # constante (ou quase)
        bottom = arr.min() - padding
        top    = arr.max() + padding
    else:
        z = (arr - mean) / std
        filt = arr[z <= z_threshold]
        if filt.size == 0:         # tudo outlier -> usa valores originais
            filt = arr
        bottom = filt.min() - padding
        top    = filt.max() + padding

    if non_negative and bottom < 0:
        bottom = 0.0

    return bottom, top



class Model:
    """
    Manages training, evaluation, and testing of neural networks for solving differential equations.

    Attributes:
        configuration (Configuration): Stores neural network and training settings.
        system_equations (SystemEquations): Defines the differential equations to solve.
        model (EquationsModel): The neural networks used to approximate solutions.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
    """
    
    def __init__(self, name: str, configuration: Configuration, system_equations: SystemEquations):
        
        self.configuration = configuration
        set_all_seeds(configuration.seed)
        self.output_path = os.path.join(OUTPUT_PATH, name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.system_equations = system_equations
        self.name = name
        
        # Initialize the neural network model
        self.model = EquationsModel(system_equations.functions, configuration.features, configuration.activation_function)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=configuration.learning_rate)

    def load(self, model_path: str, optimizer_path: str):
        """
        Loads a pre-trained model and optimizer state.
        Args:
        model_path (str): Path to the saved model weights.
        optimizer_path (str): Path to the saved optimizer state.
        """
        self.model.load_state_dict(th.load(model_path, weights_only=True))
        self.optimizer.load_state_dict(th.load(optimizer_path, weights_only=True))

    def _save_train(self, losses: List[float], best_state: dict | None = None):
        """
        Saves the trained model, optimizer state, training configuration, and loss curve.
        Args:
            losses (List[float]): List of loss values recorded during training.
        """
        assert_directory_exists(self.output_path)
        th.save(self.model.state_dict(), os.path.join(self.output_path, "model.pt"))
        th.save(self.optimizer.state_dict(), os.path.join(self.output_path, "optimizer.pt"))
        
        if best_state is not None:
            th.save(best_state["model"], os.path.join(self.output_path, "model_best.pt"))
            th.save(best_state["optimizer"], os.path.join(self.output_path, "optimizer_best.pt"))

        self.model.load_state_dict(best_state["model"])
        self.model.eval()

        t_grid = th.linspace(self.system_equations.domain[0], self.system_equations.domain[1], self.configuration.steps, dtype=th.float64).view(-1, 1).requires_grad_()
        nn_out   = self.model(t_grid)
        trial    = self.system_equations.calculate_trial_solution(t_grid, nn_out)
        x_trial = trial["x"].detach().cpu().numpy().ravel()

        ref = self.system_equations.solve_numerically(t_grid.detach().cpu().numpy().ravel() )
        x_ref = ref["x"].ravel()

        l2_error  = float(np.linalg.norm(x_trial - x_ref) / np.linalg.norm(x_ref))
        
        with open(os.path.join(self.output_path, "configuration.json"), "w") as f:
            config = self.configuration._asdict()
            config["system_equations"] = self.system_equations.configuration()
            config["system_equations"]["name"] = self.system_equations.__class__.__name__
            config["activation_function"] = config["activation_function"].__class__.__name__
            config["system_equations"]["initial_conditions"] = {
                k: (v[0], float(v[1])) for k, v in config["system_equations"]["initial_conditions"].items()}
            if best_state is not None:
                config["final_losses"] = {
                    "total"      : best_state["loss_total"],
                    "dynamics"   : best_state["loss_dynamics"],
                    "hamiltonian": best_state["loss_hamiltonian"],
                    "epoch"      : best_state["epoch"],
                }
            config["l2_error"] = l2_error
            json.dump(config, f, indent=4)

        losses_np = np.asarray(losses, dtype=float)
        smooth = smooth_curve(losses_np)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(losses_np, color="grey", alpha=0.25, label="Training Loss")
        ax.plot(np.arange(len(smooth)), smooth, lw=2, label=f"{SMOOTH.upper()}({WINDOW})")
        ax.set_title(f"Loss During Training: {self.name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        bottom, top = calculate_y_limits_log(losses, lower_pct=2, upper_pct=98, headroom=1.15)
        ax.set_yscale("symlog", linthresh=1e-5, linscale=1.0, base=10)
        ax.set_ylim(-top if bottom < 0 else max(bottom, 1e-5/10), top*1.2)
        ax.grid(True, which="both", ls=":")
        ax.legend()
        plt.savefig(os.path.join(self.output_path, "loss.png"))
        plt.show()

    def train(self):
        """
        Trains the neural network to approximate the solution of the differential equation.
        """
        assert_directory_exists(self.output_path)
        
        inputs = th.linspace(self.system_equations.domain[0], 
                             self.system_equations.domain[1], 
                             self.configuration.steps).view(-1,1).requires_grad_()

        total_losses: List[float] = []
        dyn_losses: List[float] = []
        ham_losses: List[float] = []

        best_loss = float("inf")
        best_state: Optional[dict] = None

        for _ in tqdm(range(self.configuration.epochs)):
            self.optimizer.zero_grad()
            out = self.model(inputs)

            # unpack whatever calculate_loss returned
            loss_out = self.system_equations.calculate_loss(inputs, out)
            loss, dyn, ham = loss_out["total"], loss_out["dynamics"], loss_out["hamiltonian"]
            
            total_losses.append(loss.item())
            dyn_losses.append(dyn.item())
            ham_losses.append(ham.item())

            if _ % 100 == 0:
                print(loss.item())
                
            loss.backward()
            self.optimizer.step()
            if loss.item() < best_loss:
                best_loss  = loss.item()
                best_state = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': _,
                    'loss_total': float(loss.item()),
                    'loss_dynamics': float(dyn.item()) / self.configuration.steps,
                    'loss_hamiltonian': float(ham.item()) / self.configuration.steps,
                }

        self._total_losses = np.array(total_losses)
        self._dyn_losses   = np.array(dyn_losses) / self.configuration.steps
        self._ham_losses   = np.array(ham_losses) / self.configuration.steps
            
        self._save_train(total_losses, best_state)

    def eval(self, inputs: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluates the trained neural network by computing trial solutions."""
        tensor_inputs = th.tensor(inputs, dtype=th.float64).view(-1, 1).requires_grad_()
        outputs = self.model(tensor_inputs)
        trial_solutions = self.system_equations.calculate_trial_solution(tensor_inputs, outputs)
        return {f: v.detach().numpy() for f, v in trial_solutions.items()}

    def test5(self, inputs=None):
        if inputs is None:
            inputs = np.linspace(*self.system_equations.domain,
                                 self.configuration.steps)

        filtered_inputs_t = 4 * np.arctanh(inputs)
        inputs_t = filtered_inputs_t[(filtered_inputs_t > -10) & (filtered_inputs_t < 10)]

        n = len(inputs_t)
        start = (len(inputs) - n) // 2
        end = start + n
        inputs_trimmed = inputs[start:end]
        inputs_trimmed_t = 4 * np.arctanh(inputs_trimmed)

        pred = self.eval(inputs)
        pred2 = self.eval(inputs_trimmed)
        real = self.system_equations.solve_numerically(inputs_t)
        real_comp = self.system_equations.solve_numerically(inputs)

        x_nn = pred["x"].ravel()
        y_nn = pred["y"].ravel()
        z_nn = pred["z"].ravel()
        w_nn = pred["w"].ravel()

        print(np.max(w_nn-(3 * np.pi * z_nn + np.pi * np.sin(np.pi * x_nn))))

        x2_nn = pred2["x"].ravel()

        # 3) δx(t)
        delta_x = x_nn - real_comp["x"]

        # 4) build a 3×2 grid with GridSpec
        fig = plt.figure(figsize=(14, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        ax_mse = fig.add_subplot(gs[0, 0])  # spans rows 0–1, col 0
        #ax_h   = fig.add_subplot(gs[0,    1])  # row 0, col 1
        ax_dx  = fig.add_subplot(gs[0,    1])  # row 1, col 1
        ax_xt  = fig.add_subplot(gs[1,    0])  # row 2, col 0
        ax_ps  = fig.add_subplot(gs[1,    1])  # row 2, col 1

        # — (0:0) MSE dynamics & Hamiltonian vs epochs —
        dyn_smooth = smooth_curve(self._dyn_losses)
        #ham_smooth = smooth_curve(self._ham_losses)
        epochs = np.arange(len(dyn_smooth))
        #epochs = np.arange(len(self._dyn_losses))
        #ax_mse.plot(epochs, self._dyn_losses, label="MSE Dynamics")
        #ax_mse.plot(epochs, self._ham_losses, label="MSE Hamiltonian")
        # plot raw (optional, with low alpha)…

        dyn_color = "#02FFFF"      # light-blue
        ham_color = "darkmagenta" #"#FF00FF"      # purplish
        H_color   = "purple"
        H0_color  = "#02FF00"      # vivid green
        nn_color = "#0000FF"
        zero_color = "#02FF00"
        num_color   = "red"

        ax_mse.plot(np.arange(len(self._dyn_losses)), self._dyn_losses, color=dyn_color, alpha=0.1)
        #ax_mse.plot(np.arange(len(self._ham_losses)), self._ham_losses, color=ham_color, alpha=0.1)
        ax_mse.plot(epochs, dyn_smooth, linewidth=2, label="Dynamics Loss (smoothed)", color=dyn_color)
        #ax_mse.plot(epochs, ham_smooth, linewidth=2, label="Hamiltonian Loss (smoothed)", color=ham_color)
        ax_mse.set_yscale("log")
        ax_mse.set_xlabel("Epochs")
        ax_mse.set_ylabel("Loss")
        #ax_mse.legend()

        # — (0:1) Hamiltonian vs H₀ —
        #ax_h.hlines(H0, *self.system_equations.domain, linewidth=2, color=H0_color, label="$H_0$")
        #ax_h.plot(inputs, H_nn, label="Hamiltonian (Trial Solution)", linestyle="--", linewidth=2, color=H_color)
        # clamp ±5%
        #ymin, ymax = H0 - 0.05, H0 + 0.05
        #ax_h.set_ylim(ymin, ymax)
        #ticks = [H0 - 0.05, H0, H0 + 0.05]
        #ax_h.set_yticks(ticks)
        #ax_h.set_yticklabels([r'$H_0 - 0.05$', r'$H_0$', r'$H_0 + 0.05$'])
        #ax_h.set_xlabel("Time")
        #ax_h.set_ylabel("Energy")
        #ax_h.legend()

        # — (1:1) δx(t) vs time —
        ax_dx.hlines(0, *self.system_equations.domain, color = zero_color , linewidth=2)
        ax_dx.plot(inputs, delta_x, linewidth=2, color=nn_color, linestyle="--")
        ymin, ymax = -0.25, 0.25
        ax_dx.set_ylim(ymin, ymax)
        ticks = [-0.25, 0, 0.25]
        ax_dx.set_yticks(ticks)
        ax_dx.set_xlabel("Time")
        ax_dx.set_ylabel(r"$\delta x(t)$")

        # — (2:0) x(t) NN vs numeric —
        if real is not None:
            ax_xt.plot(inputs_t, real["x"], label="Ground Truth", color=num_color, linewidth=2)
        ax_xt.plot(inputs_trimmed_t, x2_nn, label="Trial Solution", linestyle="--", linewidth=2, color=nn_color)
        ax_xt.set_xlabel("Time")
        ax_xt.set_ylabel("$x(t)$")
        #ax_xt.legend()

        # — (2:1) phase-space —
        if real is not None:
            ax_ps.plot(real["x"], real["y"], color=num_color, linewidth=2)
        ax_ps.plot(x_nn, y_nn, linestyle="--", linewidth=2, color=nn_color)
        ax_ps.set_xlabel("$x(t)$")
        ax_ps.set_ylabel("$\dot{x}(t)$")
        #ax_ps.legend()

        # — white backboard, no grids —
        for ax in (ax_mse, ax_dx, ax_xt, ax_ps):
            ax.grid(False)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for ax in (ax_mse, ax_dx, ax_xt, ax_ps):
            # Thicker spines
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            # Thicker ticks
            ax.tick_params(which='both', width=1.5, direction='in', length=6, labelsize='medium')

        handles, labels = [], []
        for ax in (ax_mse, ax_dx, ax_xt, ax_ps):
            for line in ax.get_lines():
                lab = line.get_label()
                if lab.startswith("_"):      # skip internal artists
                    continue
                if lab not in labels:
                    handles.append(line)
                    labels.append(lab)

        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.48, 0.99), fontsize="16", frameon=True)


        # save and show
        plt.savefig(os.path.join(self.output_path, "combined_test_plots.pdf"))
        plt.show()


    def test4(self, inputs=None):
        if inputs is None:
            inputs = np.linspace(*self.system_equations.domain,
                                 self.configuration.steps)

        # 1) NN vs numeric
        pred = self.eval(inputs)                            # {"x":…, "y":…}
        real = self.system_equations.solve_numerically(inputs)

        # 2) Hamiltonian curve
        x_nn = pred["x"].ravel()
        y_nn = pred["y"].ravel()
        #H_nn = 0.5 * y_nn**2 + (1 - np.cos(x_nn))
        #H0   = H_nn[0]

        # 3) δx(t)
        delta_x = x_nn - real["x"]

        # 4) build a 3×2 grid with GridSpec
        fig = plt.figure(figsize=(14, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        ax_mse = fig.add_subplot(gs[0, 0])  # spans rows 0–1, col 0
        #ax_h   = fig.add_subplot(gs[0,    1])  # row 0, col 1
        ax_dx  = fig.add_subplot(gs[0,    1])  # row 1, col 1
        ax_xt  = fig.add_subplot(gs[1,    0])  # row 2, col 0
        ax_ps  = fig.add_subplot(gs[1,    1])  # row 2, col 1

        # — (0:0) MSE dynamics & Hamiltonian vs epochs —
        dyn_smooth = smooth_curve(self._dyn_losses)
        #ham_smooth = smooth_curve(self._ham_losses)
        epochs = np.arange(len(dyn_smooth))
        #epochs = np.arange(len(self._dyn_losses))
        #ax_mse.plot(epochs, self._dyn_losses, label="MSE Dynamics")
        #ax_mse.plot(epochs, self._ham_losses, label="MSE Hamiltonian")
        # plot raw (optional, with low alpha)…

        dyn_color = "#02FFFF"      # light-blue
        ham_color = "darkmagenta" #"#FF00FF"      # purplish
        H_color   = "purple"
        H0_color  = "#02FF00"      # vivid green
        nn_color = "#0000FF"
        zero_color = "#02FF00"
        num_color   = "red"

        ax_mse.plot(np.arange(len(self._dyn_losses)), self._dyn_losses, color=dyn_color, alpha=0.1)
        #ax_mse.plot(np.arange(len(self._ham_losses)), self._ham_losses, color=ham_color, alpha=0.1)
        ax_mse.plot(epochs, dyn_smooth, linewidth=2, label="Dynamics Loss (smoothed)", color=dyn_color)
        #ax_mse.plot(epochs, ham_smooth, linewidth=2, label="Hamiltonian Loss (smoothed)", color=ham_color)
        ax_mse.set_yscale("log")
        ax_mse.set_xlabel("Epochs")
        ax_mse.set_ylabel("Loss")
        #ax_mse.legend()

        # — (0:1) Hamiltonian vs H₀ —
        #ax_h.hlines(H0, *self.system_equations.domain, linewidth=2, color=H0_color, label="$H_0$")
        #ax_h.plot(inputs, H_nn, label="Hamiltonian (Trial Solution)", linestyle="--", linewidth=2, color=H_color)
        # clamp ±5%
        #ymin, ymax = H0 - 0.05, H0 + 0.05
        #ax_h.set_ylim(ymin, ymax)
        #ticks = [H0 - 0.05, H0, H0 + 0.05]
        #ax_h.set_yticks(ticks)
        #ax_h.set_yticklabels([r'$H_0 - 0.05$', r'$H_0$', r'$H_0 + 0.05$'])
        #ax_h.set_xlabel("Time")
        #ax_h.set_ylabel("Energy")
        #ax_h.legend()

        # — (1:1) δx(t) vs time —
        ax_dx.hlines(0, *self.system_equations.domain, color = zero_color , linewidth=2)
        ax_dx.plot(inputs, delta_x, linewidth=2, color=nn_color, linestyle="--")
        ymin, ymax = -0.25, 0.25
        ax_dx.set_ylim(ymin, ymax)
        ticks = [-0.25, 0, 0.25]
        ax_dx.set_yticks(ticks)
        ax_dx.set_xlabel("Time")
        ax_dx.set_ylabel(r"$\delta x(t)$")

        # — (2:0) x(t) NN vs numeric —
        if real is not None:
            ax_xt.plot(inputs_t, real["x"], label="Ground Truth", color=num_color, linewidth=2)
        ax_xt.plot(inputs_trimmed, x_nn, label="Trial Solution", linestyle="--", linewidth=2, color=nn_color)
        ax_xt.set_xlabel("Time")
        ax_xt.set_ylabel("$x(t)$")
        #ax_xt.legend()

        # — (2:1) phase-space —
        if real is not None:
            ax_ps.plot(real["x"], real["y"], color=num_color, linewidth=2)
        ax_ps.plot(x_nn, y_nn, linestyle="--", linewidth=2, color=nn_color)
        ax_ps.set_xlabel("$x(t)$")
        ax_ps.set_ylabel("$\dot{x}(t)$")
        #ax_ps.legend()

        # — white backboard, no grids —
        for ax in (ax_mse, ax_dx, ax_xt, ax_ps):
            ax.grid(False)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for ax in (ax_mse, ax_dx, ax_xt, ax_ps):
            # Thicker spines
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            # Thicker ticks
            ax.tick_params(which='both', width=1.5, direction='in', length=6, labelsize='medium')

        handles, labels = [], []
        for ax in (ax_mse, ax_dx, ax_xt, ax_ps):
            for line in ax.get_lines():
                lab = line.get_label()
                if lab.startswith("_"):      # skip internal artists
                    continue
                if lab not in labels:
                    handles.append(line)
                    labels.append(lab)

        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.48, 0.99), fontsize="16", frameon=True)


        # save and show
        plt.savefig(os.path.join(self.output_path, "combined_test_plots.pdf"))
        plt.show()

    
    
    
    def test3(self, inputs=None):
        if inputs is None:
            inputs = np.linspace(*self.system_equations.domain,
                                 self.configuration.steps)

        # 1) NN vs numeric
        pred = self.eval(inputs)                            # {"x":…, "y":…}
        real = self.system_equations.solve_numerically(inputs)

        # 2) Hamiltonian curve
        x_nn = pred["x"].ravel()
        y_nn = pred["y"].ravel()
        H_nn = 0.5 * y_nn**2 + (1 - np.cos(x_nn))
        H0   = H_nn[0]

        # 3) δx(t)
        delta_x = x_nn - real["x"]

        # 4) build a 3×2 grid with GridSpec
        fig = plt.figure(figsize=(14, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2])

        ax_mse = fig.add_subplot(gs[0:2, 0])  # spans rows 0–1, col 0
        ax_h   = fig.add_subplot(gs[0,    1])  # row 0, col 1
        ax_dx  = fig.add_subplot(gs[1,    1])  # row 1, col 1
        ax_xt  = fig.add_subplot(gs[2,    0])  # row 2, col 0
        ax_ps  = fig.add_subplot(gs[2,    1])  # row 2, col 1

        # — (0:0) MSE dynamics & Hamiltonian vs epochs —
        dyn_smooth = smooth_curve(self._dyn_losses)
        ham_smooth = smooth_curve(self._ham_losses)
        epochs = np.arange(len(dyn_smooth))
        #epochs = np.arange(len(self._dyn_losses))
        #ax_mse.plot(epochs, self._dyn_losses, label="MSE Dynamics")
        #ax_mse.plot(epochs, self._ham_losses, label="MSE Hamiltonian")
        # plot raw (optional, with low alpha)…

        dyn_color = "#02FFFF"      # light-blue
        ham_color = "darkmagenta" #"#FF00FF"      # purplish
        H_color   = "purple"
        H0_color  = "#02FF00"      # vivid green
        nn_color = "#0000FF"
        zero_color = "#02FF00"
        num_color   = "red"

        ax_mse.plot(np.arange(len(self._dyn_losses)), self._dyn_losses, color=dyn_color, alpha=0.1)
        ax_mse.plot(np.arange(len(self._ham_losses)), self._ham_losses, color=ham_color, alpha=0.1)
        ax_mse.plot(epochs, dyn_smooth, linewidth=2, label="Dynamics Loss (smoothed)", color=dyn_color)
        ax_mse.plot(epochs, ham_smooth, linewidth=2, label="Hamiltonian Loss (smoothed)", color=ham_color)
        ax_mse.set_yscale("log")
        ax_mse.set_xlabel("Epochs")
        ax_mse.set_ylabel("Loss")
        #ax_mse.legend()

        # — (0:1) Hamiltonian vs H₀ —
        ax_h.hlines(H0, *self.system_equations.domain, linewidth=2, color=H0_color, label="$H_0$")
        ax_h.plot(inputs, H_nn, label="Hamiltonian (Trial Solution)", linestyle="--", linewidth=2, color=H_color)
        # clamp ±5%
        ymin, ymax = H0 - 0.05, H0 + 0.05
        ax_h.set_ylim(ymin, ymax)
        ticks = [H0 - 0.05, H0, H0 + 0.05]
        ax_h.set_yticks(ticks)
        ax_h.set_yticklabels([r'$H_0 - 0.05$', r'$H_0$', r'$H_0 + 0.05$'])
        ax_h.set_xlabel("Time")
        ax_h.set_ylabel("Energy")
        #ax_h.legend()

        # — (1:1) δx(t) vs time —
        ax_dx.hlines(0, *self.system_equations.domain, color = zero_color , linewidth=2)
        ax_dx.plot(inputs, delta_x, linewidth=2, color=nn_color, linestyle="--")
        ymin, ymax = -0.25, 0.25
        ax_dx.set_ylim(ymin, ymax)
        ticks = [-0.25, 0, 0.25]
        ax_dx.set_yticks(ticks)
        ax_dx.set_xlabel("Time")
        ax_dx.set_ylabel(r"$\delta x(t)$")

        # — (2:0) x(t) NN vs numeric —
        if real is not None:
            ax_xt.plot(inputs, real["x"], label="Ground Truth", color=num_color, linewidth=2)
        ax_xt.plot(inputs, x_nn, label="Trial Solution", linestyle="--", linewidth=2, color=nn_color)
        ax_xt.set_xlabel("Time")
        ax_xt.set_ylabel("$x(t)$")
        #ax_xt.legend()

        # — (2:1) phase-space —
        if real is not None:
            ax_ps.plot(real["x"], real["y"], color=num_color, linewidth=2)
        ax_ps.plot(x_nn, y_nn, linestyle="--", linewidth=2, color=nn_color)
        ax_ps.set_xlabel("$x(t)$")
        ax_ps.set_ylabel("$\dot{x}(t)$")
        #ax_ps.legend()

        # — white backboard, no grids —
        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            ax.grid(False)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            # Thicker spines
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            # Thicker ticks
            ax.tick_params(which='both', width=1.5, direction='in', length=6, labelsize='medium')

        handles, labels = [], []
        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            for line in ax.get_lines():
                lab = line.get_label()
                if lab.startswith("_"):      # skip internal artists
                    continue
                if lab not in labels:
                    handles.append(line)
                    labels.append(lab)

        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.48, 0.99), fontsize="16", frameon=True)


        # save and show
        plt.savefig(os.path.join(self.output_path, "combined_test_plots.pdf"))
        plt.show()

    

    def test2(self, inputs=None):
        if inputs is None:
            inputs = np.linspace(*self.system_equations.domain,
                                 self.configuration.steps)

        # 1) NN vs numeric
        pred = self.eval(inputs)                            # {"x":…, "y":…}
        real = self.system_equations.solve_numerically(inputs)

        # 2) Hamiltonian curve
        x_nn = pred["x"].ravel()
        y_nn = pred["y"].ravel()
        H_nn = 0.5 * y_nn**2 - (0.5*x_nn**2 - 0.25*x_nn**4)
        H0   = H_nn[0]

        # 3) δx(t)
        delta_x = x_nn - real["x"]

        # 4) build a 3×2 grid with GridSpec
        fig = plt.figure(figsize=(14, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2])

        ax_mse = fig.add_subplot(gs[0:2, 0])  # spans rows 0–1, col 0
        ax_h   = fig.add_subplot(gs[0,    1])  # row 0, col 1
        ax_dx  = fig.add_subplot(gs[1,    1])  # row 1, col 1
        ax_xt  = fig.add_subplot(gs[2,    0])  # row 2, col 0
        ax_ps  = fig.add_subplot(gs[2,    1])  # row 2, col 1

        # — (0:0) MSE dynamics & Hamiltonian vs epochs —
        dyn_smooth = smooth_curve(self._dyn_losses)
        ham_smooth = smooth_curve(self._ham_losses)
        epochs = np.arange(len(dyn_smooth))
        #epochs = np.arange(len(self._dyn_losses))
        #ax_mse.plot(epochs, self._dyn_losses, label="MSE Dynamics")
        #ax_mse.plot(epochs, self._ham_losses, label="MSE Hamiltonian")
        # plot raw (optional, with low alpha)…

        dyn_color = "#02FFFF"      # light-blue
        ham_color = "darkmagenta" #"#FF00FF"      # purplish
        H_color   = "purple"
        H0_color  = "#02FF00"      # vivid green
        nn_color = "#0000FF"
        zero_color = "#02FF00"
        num_color   = "red"

        ax_mse.plot(np.arange(len(self._dyn_losses)), self._dyn_losses, color=dyn_color, alpha=0.1)
        ax_mse.plot(np.arange(len(self._ham_losses)), self._ham_losses, color=ham_color, alpha=0.1)
        ax_mse.plot(epochs, dyn_smooth, linewidth=2, label="Dynamics Loss (smoothed)", color=dyn_color)
        ax_mse.plot(epochs, ham_smooth, linewidth=2, label="Hamiltonian Loss (smoothed)", color=ham_color)
        ax_mse.set_yscale("log")
        ax_mse.set_xlabel("Epochs")
        ax_mse.set_ylabel("Loss")
        #ax_mse.legend()

        # — (0:1) Hamiltonian vs H₀ —
        ax_h.hlines(H0, *self.system_equations.domain, linewidth=2, color=H0_color, label="$H_0$")
        ax_h.plot(inputs, H_nn, label="Hamiltonian (Trial Solution)", linestyle="--", linewidth=2, color=H_color)
        # clamp ±5%
        ymin, ymax = H0 - 0.05, H0 + 0.05
        ax_h.set_ylim(ymin, ymax)
        ticks = [H0 - 0.05, H0, H0 + 0.05]
        ax_h.set_yticks(ticks)
        ax_h.set_yticklabels([r'$H_0 - 0.05$', r'$H_0$', r'$H_0 + 0.05$'])
        ax_h.set_xlabel("Time")
        ax_h.set_ylabel("Energy")
        #ax_h.legend()

        # — (1:1) δx(t) vs time —
        ax_dx.hlines(0, *self.system_equations.domain, color = zero_color , linewidth=2)
        ax_dx.plot(inputs, delta_x, linewidth=2, color=nn_color, linestyle="--")
        ymin, ymax = -0.25, 0.25
        ax_dx.set_ylim(ymin, ymax)
        ticks = [-0.25, 0, 0.25]
        ax_dx.set_yticks(ticks)
        ax_dx.set_xlabel("Time")
        ax_dx.set_ylabel(r"$\delta x(t)$")

        # — (2:0) x(t) NN vs numeric —
        if real is not None:
            ax_xt.plot(inputs, real["x"], label="Ground Truth", color=num_color, linewidth=2)
        ax_xt.plot(inputs, x_nn, label="Trial Solution", linestyle="--", linewidth=2, color=nn_color)
        ax_xt.set_xlabel("Time")
        ax_xt.set_ylabel("$x(t)$")
        #ax_xt.legend()

        # — (2:1) phase-space —
        if real is not None:
            ax_ps.plot(real["x"], real["y"], color=num_color, linewidth=2)
        ax_ps.plot(x_nn, y_nn, linestyle="--", linewidth=2, color=nn_color)
        ax_ps.set_xlabel("$x(t)$")
        ax_ps.set_ylabel("$\dot{x}(t)$")
        #ax_ps.legend()

        # — white backboard, no grids —
        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            ax.grid(False)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            # Thicker spines
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            # Thicker ticks
            ax.tick_params(which='both', width=1.5, direction='in', length=6, labelsize='medium')

        handles, labels = [], []
        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            for line in ax.get_lines():
                lab = line.get_label()
                if lab.startswith("_"):      # skip internal artists
                    continue
                if lab not in labels:
                    handles.append(line)
                    labels.append(lab)

        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.48, 0.99), fontsize="16", frameon=True)


        # save and show
        plt.savefig(os.path.join(self.output_path, "combined_test_plots.pdf"))
        plt.show()
        
    def test1(self, inputs=None):
        if inputs is None:
            inputs = np.linspace(*self.system_equations.domain,
                                 self.configuration.steps)

        # 1) NN vs numeric
        pred = self.eval(inputs)                            # {"x":…, "y":…}
        real = self.system_equations.solve_numerically(inputs)

        # 2) Hamiltonian curve
        x_nn = pred["x"].ravel()
        y_nn = pred["y"].ravel()
        H_nn = 0.5 * y_nn**2 + (0.5*x_nn**2 - 0.25*x_nn**4)
        H0   = H_nn[0]

        # 3) δx(t)
        delta_x = x_nn - real["x"]

        # 4) build a 3×2 grid with GridSpec
        fig = plt.figure(figsize=(14, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2])

        ax_mse = fig.add_subplot(gs[0:2, 0])  # spans rows 0–1, col 0
        ax_h   = fig.add_subplot(gs[0,    1])  # row 0, col 1
        ax_dx  = fig.add_subplot(gs[1,    1])  # row 1, col 1
        ax_xt  = fig.add_subplot(gs[2,    0])  # row 2, col 0
        ax_ps  = fig.add_subplot(gs[2,    1])  # row 2, col 1

        # — (0:0) MSE dynamics & Hamiltonian vs epochs —
        dyn_smooth = smooth_curve(self._dyn_losses)
        ham_smooth = smooth_curve(self._ham_losses)
        epochs = np.arange(len(dyn_smooth))
        #epochs = np.arange(len(self._dyn_losses))
        #ax_mse.plot(epochs, self._dyn_losses, label="MSE Dynamics")
        #ax_mse.plot(epochs, self._ham_losses, label="MSE Hamiltonian")
        # plot raw (optional, with low alpha)…

        dyn_color = "#02FFFF"      # light-blue
        ham_color = "darkmagenta" #"#FF00FF"      # purplish
        H_color   = "purple"
        H0_color  = "#02FF00"      # vivid green
        nn_color = "#0000FF"
        zero_color = "#02FF00"
        num_color   = "red"

        ax_mse.plot(np.arange(len(self._dyn_losses)), self._dyn_losses, color=dyn_color, alpha=0.1)
        ax_mse.plot(np.arange(len(self._ham_losses)), self._ham_losses, color=ham_color, alpha=0.1)
        ax_mse.plot(epochs, dyn_smooth, linewidth=2, label="Dynamics Loss (smoothed)", color=dyn_color)
        ax_mse.plot(epochs, ham_smooth, linewidth=2, label="Hamiltonian Loss (smoothed)", color=ham_color)
        ax_mse.set_yscale("log")
        ax_mse.set_xlabel("Epochs")
        ax_mse.set_ylabel("Loss")
        #ax_mse.legend()

        # — (0:1) Hamiltonian vs H₀ —
        ax_h.hlines(H0, *self.system_equations.domain, linewidth=2, color=H0_color, label="$H_0$")
        ax_h.plot(inputs, H_nn, label="Hamiltonian (Trial Solution)", linestyle="--", linewidth=2, color=H_color)
        # clamp ±5%
        ymin, ymax = H0 - 0.05, H0 + 0.05
        ax_h.set_ylim(ymin, ymax)
        ticks = [H0 - 0.05, H0, H0 + 0.05]
        ax_h.set_yticks(ticks)
        ax_h.set_yticklabels([r'$H_0 - 0.05$', r'$H_0$', r'$H_0 + 0.05$'])
        ax_h.set_xlabel("Time")
        ax_h.set_ylabel("Energy")
        #ax_h.legend()

        # — (1:1) δx(t) vs time —
        ax_dx.hlines(0, *self.system_equations.domain, color = zero_color , linewidth=2)
        ax_dx.plot(inputs, delta_x, linewidth=2, color=nn_color, linestyle="--")
        ymin, ymax = -0.25, 0.25
        ax_dx.set_ylim(ymin, ymax)
        ticks = [-0.25, 0, 0.25]
        ax_dx.set_yticks(ticks)
        ax_dx.set_xlabel("Time")
        ax_dx.set_ylabel(r"$\delta x(t)$")

        # — (2:0) x(t) NN vs numeric —
        if real is not None:
            ax_xt.plot(inputs, real["x"], label="Ground Truth", color=num_color, linewidth=2)
        ax_xt.plot(inputs, x_nn, label="Trial Solution", linestyle="--", linewidth=2, color=nn_color)
        ax_xt.set_xlabel("Time")
        ax_xt.set_ylabel("$x(t)$")
        #ax_xt.legend()

        # — (2:1) phase-space —
        if real is not None:
            ax_ps.plot(real["x"], real["y"], color=num_color, linewidth=2)
        ax_ps.plot(x_nn, y_nn, linestyle="--", linewidth=2, color=nn_color)
        ax_ps.set_xlabel("$x(t)$")
        ax_ps.set_ylabel("$\dot{x}(t)$")
        #ax_ps.legend()

        # — white backboard, no grids —
        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            ax.grid(False)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            # Thicker spines
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            # Thicker ticks
            ax.tick_params(which='both', width=1.5, direction='in', length=6, labelsize='medium')

        handles, labels = [], []
        for ax in (ax_mse, ax_h, ax_dx, ax_xt, ax_ps):
            for line in ax.get_lines():
                lab = line.get_label()
                if lab.startswith("_"):      # skip internal artists
                    continue
                if lab not in labels:
                    handles.append(line)
                    labels.append(lab)

        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.48, 0.99), fontsize="16", frameon=True)


        # save and show
        plt.savefig(os.path.join(self.output_path, "combined_test_plots.pdf"))
        plt.show()
'''
def test(self, inputs: Optional[np.ndarray] = None):
        """
            Tests the trained neural network by comparing it against a numerical solution (RK-4).
            This method computes the numerical solution, evaluates the neural network's predictions and 
            plots both functions against time and phase space. It stores everything.
        Args:
            inputs (Optional[np.ndarray]): Array of time points for evaluation. If 'None', a uniform grid is used.
        """
        if inputs is None:
            inputs = np.linspace(self.system_equations.domain[0], self.system_equations.domain[1], self.configuration.steps)

        predicted_values = self.eval(inputs)
        real_values = self.system_equations.solve_numerically(inputs)
        assert_directory_exists(self.output_path)
      
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        if real_values is not None:
            axes[0].plot(inputs, real_values["x"], label="Numerical Solution: x(t)", color="red")
        axes[0].plot(inputs, predicted_values["x"], label="NN Approximation: x(t)", color="darkblue", linestyle=(0, (3, 3)))
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("x(t)")
        axes[0].legend()
        
        if real_values is not None:
            axes[1].plot(real_values["x"], real_values["y"], label="Numerical Phase Space", color="red")
        axes[1].plot(predicted_values["x"], predicted_values["y"], label="NN Phase Space", color="darkblue", linestyle=(0, (3, 3)))
        axes[1].set_xlabel("x(t)")
        axes[1].set_ylabel("x'(t)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "solution.png"))
        plt.show()'''
