from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    """
    central difference is more accurate, as Taylor's theorem goes, is O(h^2),
    while forward and backward difference are O(h)
    """
    vals_ls = list(vals)
    vals_plus = vals_ls.copy()
    vals_plus[arg] += epsilon / 2

    vals_minus = vals_ls.copy()
    vals_minus[arg] -= epsilon / 2

    f_plus, f_minus = f(*vals_plus), f(*vals_minus)

    return (f_plus - f_minus) / epsilon
    raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # assume the graph is a DAG, no assert now!!!
    def dfs(u: Variable):
        # print(u, u.is_constant(), u.is_leaf())
        if u.is_constant():
            return []
        ret = [u]
        """
        With example of `y = w2*(w1*x + b1)`
        parents of `y` are `[w2*(w1*x + b1), b2]`
        parents of `w2*(w1*x + b1)` are `[w2, (w1*x + b1)]`
        so it return at `w2` but continue at (w1*x + b1) as the same
        that's why, it's called `leaf`.
        """
        if u.is_leaf():
            return ret
        for v in u.parents:
            ret.extend(dfs(v))
        return ret

    return dfs(variable)
    raise NotImplementedError("Need to implement for Task 1.4")


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    topo_ls = topological_sort(variable)
    # print("topo_ls")
    # print(variable, len(topo_ls))
    deriv_dict = {variable.unique_id: deriv}
    for u in topo_ls:
        if u.unique_id is not variable.unique_id and u.is_leaf():
            continue
        d = deriv_dict[u.unique_id]
        for input, dd in u.chain_rule(d):
            if input.is_leaf():
                input.accumulate_derivative(dd)
            if input.unique_id not in deriv_dict:
                deriv_dict[input.unique_id] = 0.0
            deriv_dict[input.unique_id] = dd
            # print(deriv_dict[input.unique_id])
        # print(u, d, u.derivative)
    return
    raise NotImplementedError("Need to implement for Task 1.4")


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
