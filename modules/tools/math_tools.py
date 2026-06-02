import ast
import math
import operator

from modules.tools.registry import registry


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}

_MAX_EXPONENT = 1000


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported operator")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError("unsupported unary operator")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"unknown name '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("unknown function")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError("unsupported expression")


def _format_number(x) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return repr(round(x, 10))
    return str(x)


@registry.register({
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic expression and return the numeric result. "
            "Supports + - * / // % ** , parentheses, unary minus, the functions "
            "sqrt, sin, cos, tan, log, log10, exp, floor, ceil, abs, round, min, max, "
            "and the constants pi, e, tau. Trig functions take radians. "
            "Use this for any arithmetic the user asks for, e.g. '15 * 240 / 100' or 'sqrt(144) + pi'."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The arithmetic expression to evaluate, e.g. '2 + 2 * 10' or 'sqrt(16)'. "
                        "Do not include an equals sign, words, or units."
                    ),
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
})
def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
    except ZeroDivisionError:
        return "That would divide by zero."
    except Exception as e:
        return f"I couldn't evaluate that expression: {e}"
    return _format_number(result)

