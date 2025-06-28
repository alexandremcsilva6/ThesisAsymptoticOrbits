"""
Definition of the mathematical problem for solving differential equations.

This file provides:
1) Abstract base classes to define and solve systems of differential equations.
2) Trial solution constructions for neural network approximations.
3) Numerical solutions for comparison using SciPy's numerical solvers (Runge-Kutta).
"""

import abc 
from typing import Dict, Optional, List, Tuple, Any, Iterable, Union
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch as th
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import interp1d 


def simpson_integral(y: th.Tensor, x: th.Tensor) -> th.Tensor:
    """
    Composite Simpson's rule on a uniform grid.
    y, x are 1D tensors of the same length.
    If len is even, drop the last point so we have an odd number.
    Returns a scalar Tensor.
    """
    n = y.shape[0]
    # Simpson needs n odd (#points = 2m+1)
    if n % 2 == 0:
        n = n - 1
        y = y[:n]
        x = x[:n]

    h = (x[-1] - x[0]) / (n - 1)
    # endpoints
    s = y[0] + y[-1]
    # 4 * odd indices, 2 * even indices
    s = s + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()
    return s * (h / 3)

def compute_derivative(inputs: th.Tensor, outputs: th.Tensor) -> th.Tensor:
    """
    Computes the derivative of a given tensor with respect to its inputs.

    Args:
        inputs (th.Tensor): The input tensor (e.g., time points).
        outputs (th.Tensor): The output tensor (e.g., trial solution values).

    Returns:
        th.Tensor: The computed derivative of outputs with respect to inputs.
    """
    return th.autograd.grad(outputs, inputs, grad_outputs=th.ones_like(outputs), create_graph=True)[0]

def compute_nth_derivative(t: th.Tensor, y: th.Tensor, n: int) -> th.Tensor:
    """
    Recursively apply autograd to obtain the *n*-th derivative of y w.r.t t.
    """
    out = y
    for _ in range(n):
        out = th.autograd.grad(out, t, grad_outputs=th.ones_like(out), create_graph=True)[0]
    return out

class SystemEquations(abc.ABC):
    """
    Abstract base class for defining and solving systems of first-order differential equations.

    Attributes:
        functions (List[str]): List of variable names (e.g., ["x", "y"]).
        domain (Tuple[float, float]): Time domain for the equations (start, end).
        initial_conditions (Dict[str, Tuple[float, float]]): Initial conditions for each variable.
    """
    def __init__(self, functions: List[str], domain: Tuple[float, float], initial_conditions: Dict[str, Tuple[float, float]]):
        self.functions = functions
        self.domain = domain
        self.initial_conditions = initial_conditions

    def configuration(self) -> Dict[str, Any]:
        """ Returns the system configuration. """
        return {
            "functions": self.functions,
            "domain": self.domain,
            "initial_conditions": self.initial_conditions,
        }

    def calculate_trial_solution(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Constructs trial solutions that satisfy initial conditions.

        Args:
            inputs (th.Tensor): Input tensor (time values).
            outputs (Dict[str, th.Tensor]): Neural network outputs.

        Returns:
            Dict[str, th.Tensor]: Dictionary containing trial solutions for each function.
        """
        return {
           function: self.initial_conditions[function][1] + (inputs - self.initial_conditions[function][0]) * outputs[function]
            for function in self.functions
        }
    
    @abc.abstractmethod
    def calculate_loss(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Computes the loss function based on the system dynamics.
        
        Returns:
            th.Tensor: Computed loss value.
        """
        raise NotImplementedError

    def system(self, t: float, y: Union[np.ndarray, Iterable, int, float]) -> Union[np.ndarray, Iterable, int, float]:
        """
        Defines the system of first-order differential equations.

        Args:
            t (float): Time variable.
            y (Union[np.ndarray, Iterable, int, float]): State variables.

        Returns:
            Union[np.ndarray, Iterable, int, float]: The computed derivatives.
        """
        raise NotImplementedError

    def solve_numerically(self, inputs: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Solves the system of equations numerically using the `solve_ivp` function.

        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary of numerical solutions for each variable.
        """
        try:
            inputs_np = inputs.view(-1).cpu().numpy() if isinstance(inputs, th.Tensor) else inputs
            
            sol = solve_ivp(
                self.system,
                self.domain,
                [self.initial_conditions[var][1] for var in self.functions],
                t_eval=inputs_np
            )
            
            return {function: sol.y[i] for i, function in enumerate(self.functions)}
        except NotImplementedError:
            return None


class SecondOrderEquations(abc.ABC):
    """
    Abstract base class for solving second-order differential equations.

    Attributes:
        function (str): The dependent variable (e.g., "x").
        domain (Tuple[float, float]): Start and end points of the time domain.
        initial_conditions (Dict[str, Tuple[float, float]]): Initial conditions for the function and its derivative.
        boundary_type (str, optional): Type of boundary conditions to enforce ("pvi" or "dirichlet"). Default is "pvi".
    """
    def __init__(self, function: str, domain: Tuple[float, float], initial_conditions: Dict[str, Tuple[float, float]], 
                 boundary_type: str = "pvi"):
        
        self.function = function
        self.functions = [function]  # Keep as list for compatibility
        self.domain = domain
        self.initial_conditions = initial_conditions
        self.boundary_type = boundary_type  # Store boundary type

    def configuration(self) -> Dict[str, Any]:
        """
        Returns the configuration of the second-order equation.

        Returns:
            Dict[str, Any]: A dictionary containing the equation configuration.
        """
        return {
            "function": self.function,
            "domain": self.domain,
            "initial_conditions": self.initial_conditions,
            "boundary_type": self.boundary_type,
        }

    def calculate_trial_solution(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Constructs the trial solution for second-order differential equations.
        Returns:
            Dict[str, th.Tensor]: Dictionary containing trial solutions for the function and its first derivative.
        """
        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)

        # Neural network's output is N(x, params)
        N_x = outputs[self.function]

        if self.boundary_type == "pvi":
            x0, y0, t0 = self.initial_conditions["x"][1], self.initial_conditions["y"][1], self.initial_conditions["x"][0]
            x_trial = x0 + y0 * (inputs - t0) + (0.5)*(inputs - t0)**2 * N_x
        
        elif self.boundary_type == "dirichlet":
            a, b = self.domain
            y_a, y_b = self.initial_conditions["x"][1], self.initial_conditions["y"][1]
            x_trial = y_a * ((b - inputs) / (b - a)) + y_b * ((inputs - a) / (b - a)) + ((inputs - a) * (b - inputs) * N_x)
        else:
            raise ValueError("Invalid boundary_type. Use 'pvi' for Initial Value Problem or 'dirichlet' for Dirichlet Conditions.")
        
        # Compute the derivative for x'(t)
        x_dot_trial = compute_derivative(inputs, x_trial)
        return {"x": x_trial, "y": x_dot_trial}

    @abc.abstractmethod
    def calculate_loss(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> th.Tensor:
        raise NotImplementedError

    def solve_numerically(self, inputs: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Solves the second-order differential equation numerically.

        Args:
            inputs (np.ndarray): Array of time points where the solution is evaluated.

        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary of numerical solutions for "x" and "y" (x' if PVI, dx/dt if Dirichlet).
        """
        try:
            inputs_np = inputs.view(-1).cpu().numpy() if isinstance(inputs, th.Tensor) else inputs

            if self.boundary_type == "pvi":
                # Solve as an initial value problem using solve_ivp
                def system(t, z):
                    x, y = z  # y = dx/dt
                    dxdt = y
                    dydt = self.equation(x, y, t)  # Second-order ODE
                    return [dxdt, dydt]

                sol = solve_ivp(
                    system,
                    self.domain,
                    [self.initial_conditions["x"][1], self.initial_conditions["y"][1]],
                    t_eval=inputs_np
                )

                real_values = {
                    "x": interp1d(sol.t, sol.y[0], kind='cubic', fill_value="extrapolate")(inputs_np),
                    "y": interp1d(sol.t, sol.y[1], kind='cubic', fill_value="extrapolate")(inputs_np),
                }

            elif self.boundary_type == "dirichlet":
                # Solve as a boundary value problem using solve_bvp
                from scipy.integrate import solve_bvp

                a, b = self.domain  # Boundary points
                y_a = self.initial_conditions["x"][1]  # x(a)
                y_b = self.initial_conditions["y"][1]  # x(b)

                def system(t, Y):
                    x, y = Y
                    dxdt = y
                    dydt = self.equation(x, y, t)
                    return np.vstack((dxdt, dydt))

                def bc(Y_a, Y_b):
                    return np.array([Y_a[0] - y_a,  # x(a) = y_a
                                     Y_b[0] - y_b])  # x(b) = y_b

                # Initial guess for solution
                t_guess = np.linspace(a, b, 100)
                x_guess = np.linspace(y_a, y_b, 100)  # Linear interpolation
                y_guess = np.zeros_like(x_guess)  # Assume x' ≈ 0 initially
                Y_guess = np.vstack((x_guess, y_guess))

                sol = solve_bvp(system, bc, t_guess, Y_guess)

                real_values = {
                    "x": interp1d(sol.x, sol.y[0], kind='cubic', fill_value="extrapolate")(inputs_np),
                    "y": interp1d(sol.x, sol.y[1], kind='cubic', fill_value="extrapolate")(inputs_np),
                }

            else:
                raise ValueError("Invalid boundary_type. Use 'pvi' for Initial Value Problems or 'dirichlet' for Boundary Value Problems.")

            return real_values

        except NotImplementedError:
            return None


class FourthOrderEquations(abc.ABC):

    def __init__(self, function: str, domain: Tuple[float, float], initial_conditions: Dict[str, Tuple[float, float]],boundary_type: str = "dirichlet"):

        self.function = function
        self.functions = [function]
        self.domain = domain
        self.initial_conditions = initial_conditions
        self.boundary_type = boundary_type

    def configuration(self) -> Dict[str, Any]:
        return {
            "function": self.function,
            "domain": self.domain,
            "initial_conditions": self.initial_conditions,
            "boundary_type": self.boundary_type,
        }

    def calculate_trial_solution(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Build trial functions x, y, z, w ensuring the 4 IVP conditions.
        """
        if not inputs.requires_grad: 
            inputs = inputs.clone().detach().requires_grad_(True)

        t0 = self.initial_conditions["x"][0]
        N_t = outputs[self.function]

        if self.boundary_type == "pvi":
            x0, y0, z0, w0, t0 = self.initial_conditions["x"][1], self.initial_conditions["y"][1], self.initial_conditions["z"][1], self.initial_conditions["w"][1], self.initial_conditions["x"][0]
            x_trial = x0 + y0 * (inputs - t0) + 0.5*z0 * (inputs - t0)**2 + (1/6)*w0 * (inputs - t0)**3 + (1/24)*(inputs - t0)**4 * N_t

        elif self.boundary_type == "dirichlet":
            a, b = self.domain
            x_a, y_a = self.initial_conditions["x"][1], self.initial_conditions["y"][1]
            x_b, y_b = self.initial_conditions["z"][1], self.initial_conditions["w"][1]
            tau = (inputs-a)/(b-a)
            H0 = 2*tau**3 - 3*tau**2 + 1
            H1 = (b-a)*(tau**3 -2*tau**2 + tau)
            H2 = -2*tau**3 +3*tau**2
            H3 = (b-a)*(tau**3 - tau**2)
            x_trial = H0*x_a + H1*y_a + H2*x_b + H3*y_b + ((inputs - a)**2 * (b - inputs)**2 * N_t)
            
        else:
            raise ValueError("Invalid boundary_type. Use 'pvi' for Initial Value Problem or 'dirichlet' for Dirichlet Conditions (not working).")

        y_trial = compute_derivative(inputs, x_trial)
        z_trial = compute_derivative(inputs, y_trial)
        w_trial = compute_derivative(inputs, z_trial)
        return {"x": x_trial, "y": y_trial, "z": z_trial, "w": w_trial}

    @abc.abstractmethod
    def calculate_loss(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> th.Tensor:
        raise NotImplementedError

    def solve_numerically(self, inputs: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Integrates the equivalent 1st-order system with SciPy for reference. Only 'pvi' is implemented.
        """
        try:
            t0, tf = self.domain
            if self.boundary_type == "pvi":
                def system(t, Z):
                    x, y, z, w = Z
                    dxdt = y
                    dydt = z
                    dzdt = w
                    dwdt = self.equation(x, y, z, w, t)
                    return [dxdt, dydt, dzdt, dwdt]

                ic = [self.initial_conditions["x"][1], self.initial_conditions["y"][1], self.initial_conditions["z"][1], self.initial_conditions["w"][1]]
            
                sol = solve_ivp(system, (t0, tf), ic, t_eval=inputs)

                interp = lambda k: interp1d(sol.t, sol.y[k], kind="cubic", fill_value="extrapolate")(inputs)

            elif self.boundary_type == "dirichlet":

                a, b = -20, 20
                x_a = self.initial_conditions["x"][1]
                y_a = self.initial_conditions["y"][1]
                x_b = self.initial_conditions["z"][1]
                y_b = self.initial_conditions["w"][1]

                def system(t, Y):
                    x, y, z, w = Y
                    #t_phys = 4 * np.arctanh(t)
                    dxdt = y
                    dydt = z
                    dzdt = w
                    dwdt = self.equation(x, y, z, w, t)
                    return np.vstack((dxdt, dydt, dzdt, dwdt))

                def bc(Y_a, Y_b):
                    return np.array([Y_a[0] - x_a, Y_a[1] - y_a, Y_b[0] - x_b, Y_b[1] - y_b])

                # Initial guess for solution
                t_guess = np.linspace(a, b, 200)
                x_guess = np.linspace(x_a, x_b, t_guess.size)
                y_guess = np.linspace(y_a, y_b, t_guess.size)
                z_guess = np.zeros_like(x_guess)
                w_guess = np.zeros_like(x_guess)
                Y_guess = np.vstack((x_guess, y_guess, z_guess, w_guess))

                inputs_np = inputs if isinstance(inputs,np.ndarray) else inputs.detach().cpu().numpy().ravel()

                sol = solve_bvp(system, bc, t_guess, Y_guess, tol=1e-5)

                interp = lambda k: interp1d(sol.x, sol.y[k], kind="cubic", fill_value="extrapolate")(inputs_np)

            return {"x": interp(0), "y": interp(1), "z": interp(2), "w": interp(3)}

        except NotImplementedError:
            return None





