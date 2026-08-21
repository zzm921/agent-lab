"""计算器工具：用 AST 白名单安全求值，防止任意代码执行。"""
import ast
import operator

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    """仅允许数字常量、四则运算、取模、幂与括号。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的运算符")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的运算符")
        return op(_safe_eval(node.operand))
    raise ValueError("仅支持数字与四则运算表达式")


@tool
def calculator(expression: str) -> str:
    """安全计算数学表达式，支持 + - * / % ** 与括号，返回计算结果。"""
    normalized = (
        expression.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("（", "(")
        .replace("）", ")")
    )
    tree = ast.parse(normalized, mode="eval")
    result = round(float(_safe_eval(tree.body)), 6)
    if result.is_integer():
        return str(int(result))
    return str(result)
