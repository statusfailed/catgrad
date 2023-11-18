import ast
from typing import Any, List, Tuple, Callable

from catgrad.signature import Operation, Dtype, NdArrayType, obj
import catgrad.operations as ops
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

def expr(fn: Callable[Apply, ast.Expr]) -> Callable[[Apply, List[ast.Name]], List[ast.Assign]]:
    """ A decorator to turn a function of type
    ``Apply → ast.Expr``
    into one of type
    ``Apply → List[ast.Assign]``
    """
    def expr_wrapper(apply: Apply):
        if len(apply.lhs) == 0:
            return []
        elif len(apply.lhs) == 1:
            lhs = store(apply.lhs[0])
        else:
            lhs = ast.Tuple(elts=[ store(i) for i in apply.lhs ], ctx=ast.Store())

        args = [ load(i) for i in apply.rhs ]
        expr = fn(apply, args)
        return [ast.Assign(targets=[lhs], value=expr)]

    return expr_wrapper

def binop(b: ast.BinOp) -> Callable[Apply, List[ast.Assign]]:
    def binop_wrapper(a: Apply, args: List[ast.Name]) -> ast.Expr:
        assert len(args) == 2
        return ast.BinOp(left=args[0], op=b, right=args[1])
    return expr(binop_wrapper)

def _call_backend(method: str, args) -> ast.Call:
    # AST expression for calling a backend method, e.g., "self.backend.compose(x, y, z)"
    _assert_identifier(method)
    return ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='backend', ctx=ast.Load()),
            attr=method, ctx=ast.Load()),
        args=args, keywords=[])

################################################################################
# Expression handlers: functions turning Operations into python ast.Expr nodes.

@expr
def copy(a: Apply, args: List[ast.Name]) -> ast.Expr:
    assert type(a.op) == ops.Copy
    assert len(args) == 1
    return ast.Tuple(elts=args*len(a.target), ctx=ast.Load())

def discard(a: Apply) -> List[ast.Assign]:
    assert type(a.op) == ops.Discard
    assert len(a.rhs) == 1
    # Discarding produces no assignment statements.
    return []

@expr
def constant(a: Apply, args: List[ast.Name]) -> ast.Call:
    assert type(a.op) == ops.Constant
    assert len(args) == 0
    dtype = ast.Attribute(
            value=ast.Name(id="Dtype", ctx=ast.Load()),
            attr=a.op.T.dtype.name,
            ctx=ast.Load())
    return _call_backend('constant', [ast.Constant(value=a.op.x), dtype])

# An expression like self.backend.compose(x_0, x_1, 2)
@expr
def compose(a: Apply, args: List[ast.Name]) -> ast.Call:
    assert type(a.op) == ops.Compose
    assert len(args) == 2
    axes = ast.Constant(value=len(a.op.B.shape))
    return _call_backend('compose', args + [axes])

@expr
def reshape(a: Apply, args: List[ast.Name]) -> ast.Call:
    assert type(a.op) == ops.Reshape
    assert len(args) == 1
    return _call_backend('reshape', [ args[0], ast.Constant(value=a.op.Y.shape) ])

@expr
def transpose(a: Apply, args: List[ast.Name]) -> ast.Call:
    assert type(a.op) == ops.Transpose
    assert len(args) == 1
    return _call_backend('transpose', [ args[0] ])

# Handlers for each operation
# Each function here takes an Assignment ....
OP_HANDLERS: dict[Operation, Callable[Apply, List[ast.Assign]]] = {
    ops.Copy: copy,
    ops.Discard: discard,
    ops.Add: binop(ast.Add()),
    ops.Multiply: binop(ast.Mult()),
    ops.Constant: constant,
    ops.Compose: compose, # binop(ast.MatMult()), # TODO: use binop matmult if len(B) == 1?
    ops.Reshape: reshape,
    ops.Transpose: transpose,
}

################################################################################
# Helpers to make python definitions

def _mk_arguments(names: List[ast.Name]):
    return ast.arguments(args=[ ast.arg(n) for n in names ], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[])

def _mk_module(name: str, body: List[ast.FunctionDef]) -> ast.Module:
    _assert_identifier(name)
    backend_assign = ast.AnnAssign(
        target=ast.Name(id='backend', ctx=ast.Store()),
        annotation=ast.Name(id='ArrayBackend', ctx=ast.Load()),
        simple=1)

    body = [backend_assign] + body
    dc_import = ast.ImportFrom(module='dataclasses', names=[ast.alias(name='dataclass')], level=0)
    cg_import = ast.ImportFrom(module='catgrad.signature', names=[ast.alias(name='Dtype')], level=0)
    ab_import = ast.ImportFrom(module='catgrad.target.python.array_backend', names=[ast.alias(name='ArrayBackend')], level=0)
    class_def = ast.ClassDef(
            name=name,
            bases=[],
            keywords=[],
            body=body,
            decorator_list=[ast.Name(id="dataclass", ctx=ast.Load())],
            type_params=[])

    return ast.Module(body=[dc_import, cg_import, ab_import, class_def], type_ignores=[])

def _mk_function_definition(f: OpenHypergraph, name: str = 'fn', op_handlers=OP_HANDLERS) -> ast.FunctionDef:
    _assert_identifier(name)

    # Get a catgrad FunctionDefinition
    fn = FunctionDefinition.from_open_hypergraph(f)

    # create a python FunctionDef
    args = _mk_arguments(["self"] + [name_id(i) for i in fn.args])
    body = [ assignment for apply in fn.body for assignment in op_handlers[type(apply.op)](apply) ]
    rval = ast.List(elts=[ load(i) for i in fn.returns ], ctx=ast.Load())
    body.append(ast.Return(value=rval, ctx=ast.Load()))
    return ast.FunctionDef(
        name = name,
        args = args,
        body = body,
        decorator_list=[],
        type_params=[])

def to_python_class(fs: dict[str, OpenHypergraph], class_name: str = 'Dynamic'):
    fn_defs = []
    for name, f in fs.items():
        _assert_identifier(name)
        fn_defs.append(_mk_function_definition(f, name))
    mod_ast = _mk_module(class_name, fn_defs)
    ast.fix_missing_locations(mod_ast)
    env = {}
    exec(compile(mod_ast, filename='<string>', mode='exec'), env)
    return env[class_name]

def to_python_function(f: OpenHypergraph, function_name: str = 'fn', filename='<string>') -> Callable:
    # compile to a class
    Dynamic = to_python_class({function_name: f}, 'Dynamic')

    # instantiate with numpy array backend and return closure over class
    from catgrad.target.python.array_backend import Numpy
    d = Dynamic(Numpy)
    return (lambda *args: d.fn(*args))
