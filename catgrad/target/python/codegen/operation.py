""" Generate python expressions for each operation in catgrad.core.operation """
import ast
from typing import Type, List, Tuple, Callable

from catgrad.signature import Dtype
import catgrad.core.operation as ops
from catgrad.target.ast import *

################################################################################
# Helpers

def _assert_identifier(name: str):
    if not name.isidentifier():
        raise ValueError(f"name {name} is not a valid identifier")

def name_id(i: int):
    return f"x{i}"

def load(i: int):
    return ast.Name(name_id(i), ctx=ast.Load())

def store(i: int):
    return ast.Name(name_id(i), ctx=ast.Store())

def expr_lhs(apply: Apply):
    """ Return the LHS (targets) of an assignment as an AST.
    For example, when ``apply`` has lhs [x0, x1, x2], this will return a tuple
    (x0, x1, x2) with context Store.
    """
    if len(apply.lhs) == 0:
        return []
    elif len(apply.lhs) == 1:
        lhs = store(apply.lhs[0])
    else:
        lhs = ast.Tuple(elts=[ store(i) for i in apply.lhs ], ctx=ast.Store())
    return lhs

def expr(fn: Callable[[Apply, List[ast.Name]], ast.expr]) -> Callable[[Apply], List[ast.Assign]]:
    """ A decorator to turn a function of type
    ``Apply → ast.expr``
    into one of type
    ``Apply → List[ast.Assign]``
    """
    def expr_wrapper(apply: Apply):
        lhs = expr_lhs(apply)
        args = [ load(i) for i in apply.rhs ]
        expr = fn(apply, args)
        return [ast.Assign(targets=[lhs], value=expr)]

    return expr_wrapper

def binop(b) -> Callable[[Apply], List[ast.Assign]]:
    def binop_wrapper(a: Apply, args: List[ast.Name]) -> ast.expr:
        assert len(args) == 2
        return ast.BinOp(left=args[0], op=b, right=args[1])
    return expr(binop_wrapper)

def comparison(b) -> Callable[[Apply], List[ast.Assign]]:
    def binop_wrapper(a: Apply, args: List[ast.Name]) -> ast.expr:
        assert len(args) == 2
        return ast.Compare(left=args[0], ops=[b], comparators=[args[1]])
    return expr(binop_wrapper)

def _call_backend(method: str, args) -> ast.Call:
    # AST expression for calling a backend method, e.g., "self.backend.compose(x, y, z)"
    _assert_identifier(method)
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_backend', ctx=ast.Load()),
            attr=method, ctx=ast.Load()),
        args=args, keywords=[])

################################################################################
# Expression handlers: functions turning operations into python ast.expr nodes.

@expr
def copy(a: Apply, args: List[ast.Name]) -> ast.expr:
    assert type(a.op) == ops.Copy
    assert len(args) == 1
    return ast.Tuple(elts=args*len(a.target), ctx=ast.Load())

def discard(a: Apply) -> List[ast.Assign]:
    assert type(a.op) == ops.Discard
    assert len(a.rhs) == 1
    # Discarding produces no assignment statements.
    return []

@expr
def ncopy(a: Apply, args: List[ast.Name]) -> ast.expr:
    assert type(a.op) == ops.NCopy
    assert len(args) == 1
    return _call_backend('ncopy', [ast.Constant(value=a.op.T.shape), args[0]])

@expr
def nsplit(a: Apply, args: List[ast.Name]) -> ast.expr:
    assert type(a.op) == ops.NSplit
    assert len(args) == 1
    return _call_backend('nsplit', [args[0], ast.Constant(value=a.op.k)])

@expr
def nconcatenate(a: Apply, args: List[ast.Name]) -> ast.expr:
    # variadic; should have k arguments
    assert type(a.op) == ops.NConcatenate
    assert len(args) == a.op.k
    var_args = ast.List(elts=args, ctx=ast.Load())
    return _call_backend('nconcatenate', [var_args, ast.Constant(value=a.op.k)])

@expr
def nadd(a: Apply, args: List[ast.Name]) -> ast.expr:
    assert type(a.op) == ops.NAdd
    assert len(args) == 1
    # dimensions should be e.g., (-3, -2, -1) for
    dims = tuple( -(i+1) for i in reversed(range(len(a.op.T.shape))) )
    return _call_backend('nadd', [ast.Constant(value=dims), args[0]])

@expr
def nmax(a: Apply, args: List[ast.Name]) -> ast.expr:
    assert type(a.op) == ops.NMax
    assert len(args) == 1
    # dimensions should be e.g., (-3, -2, -1) for
    dims = tuple( -(i+1) for i in reversed(range(len(a.op.T.shape))) )
    return _call_backend('nmax', [ast.Constant(value=dims), args[0]])

@expr
def negate(a: Apply, args: List[ast.Name]) -> ast.expr:
    assert type(a.op) == ops.Negate
    assert len(args) == 1
    return ast.UnaryOp(op=ast.USub(), operand=args[0])

@expr
def invert(a: Apply, args: List[ast.Name]) -> ast.expr:
    assert type(a.op) == ops.Invert
    assert len(args) == 1
    return ast.UnaryOp(op=ast.Invert(), operand=args[0])

@expr
def divide(a: Apply, args: List[ast.Name]) -> ast.expr:
    assert type(a.op) == ops.Divide
    assert len(args) == 2
    b = ast.Div() if a.op.T.dtype.is_floating() else ast.FloorDiv()
    return ast.BinOp(left=args[0], op=b, right=args[1])

@expr
def constant(a: Apply, args: List[ast.Name]) -> ast.Call:
    assert type(a.op) == ops.Constant
    assert len(args) == 0
    shape = ast.Tuple(
            elts=[ ast.Constant(value=i) for i in a.op.T.shape ],
            ctx=ast.Load())
    dtype = ast.Attribute(
            value=ast.Name(id="Dtype", ctx=ast.Load()),
            attr=a.op.T.dtype.name,
            ctx=ast.Load())
    return _call_backend('constant', [ast.Constant(value=a.op.x), shape, dtype])

# An expression like self.backend.compose(x_0, x_1, 2)
@expr
def compose(a: Apply, args: List[ast.Name]) -> ast.Call:
    assert type(a.op) == ops.Compose
    assert len(args) == 2
    axes = len(a.op.B.shape)
    # note: we inline with X @ Y if matrices are of the right shape.
    if axes == 1 and len(a.op.A.shape) == 1 and len(a.op.C.shape) == 1:
        return ast.BinOp(left=args[0], op=ast.MatMult(), right=args[1])
    else:
        axes = ast.Constant(value=len(a.op.B.shape))
        return _call_backend('compose', args + [axes])

@expr
def reshape(a: Apply, args: List[ast.Name]) -> ast.Call:
    assert type(a.op) == ops.Reshape
    assert len(args) == 1
    return _call_backend('reshape', [ args[0], ast.Constant(value=a.op.Y.shape) ])

@expr
def permute(a: Apply, args: List[ast.Name]) -> ast.Call:
    assert type(a.op) == ops.Permute
    assert len(args) == 1
    # permutation array as a constant expression
    p_arg = ast.List(elts=[ast.Constant(i, ctx=ast.Load()) for i in a.op.p], ctx=ast.Load())
    return _call_backend('permute', [ args[0], p_arg ])

# Handlers for each operation
# Each function here takes an Assignment ....
OP_HANDLERS: dict[Type[operation], Callable[[Apply], List[ast.Assign]]] = {
    # core operations
    ops.Copy: copy,
    ops.NCopy: ncopy,
    ops.NSplit: nsplit,
    ops.NConcatenate: nconcatenate,
    ops.Discard: discard,
    ops.Add: binop(ast.Add()),
    ops.NAdd: nadd,
    ops.NMax: nmax,
    ops.Negate: negate,
    ops.Invert: invert,
    ops.Subtract: binop(ast.Sub()),
    ops.Multiply: binop(ast.Mult()),
    ops.Divide: divide,
    ops.Power: binop(ast.Pow()),
    ops.Constant: constant,
    ops.MatrixMultiply: binop(ast.MatMult()), # TODO: use binop matmult if len(B) == 1?
    ops.Compose: compose, # binop(ast.MatMult()), # TODO: use binop matmult if len(B) == 1?
    ops.Reshape: reshape,
    ops.Permute: permute,
    ops.Gt: comparison(ast.Gt()),
}
