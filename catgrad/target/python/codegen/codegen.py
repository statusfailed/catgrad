from typing import List, Callable
import ast

from open_hypergraphs import OpenHypergraph

from catgrad.special.definition import Definition

from catgrad.target.ast import FunctionDefinition
from catgrad.target.python.special import PythonOp, _extract_python_ops
from catgrad.target.python.codegen.operation import _assert_identifier, name_id, load, OP_HANDLERS
from catgrad.target.python.codegen.definition import _extract_definitions, definition
from catgrad.target.python.array_backend.numpy import Numpy

################################################################################
# Create python definitions

def _mk_arguments(names: List[ast.Name]):
    return ast.arguments(args=[ ast.arg(n) for n in names ], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[])

def _mk_module(name: str, fn_defs: List[ast.FunctionDef]) -> ast.Module:
    _assert_identifier(name)
    backend_assign = ast.AnnAssign(
        target=ast.Name(id='_backend', ctx=ast.Store()),
        annotation=ast.Name(id='ArrayBackend', ctx=ast.Load()),
        simple=1)

    body = [backend_assign] + fn_defs

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

def _mk_function_definition(f: OpenHypergraph, def_symbols: dict[Definition, str], name: str = 'fn', python_op_symbols=None, op_handlers=OP_HANDLERS) -> ast.FunctionDef:
    _assert_identifier(name)

    # Get a catgrad FunctionDefinition
    fn = FunctionDefinition.from_open_hypergraph(f)

    # create a python FunctionDef
    args = _mk_arguments(["self"] + [name_id(i) for i in fn.args])
    body = []
    for apply in fn.body:
        op_type = type(apply.op)

        # we handle Definition as a special case, rather than looking up how to
        # handle it in OP_HANDLERS.
        # This because we need to pass extra info - def_symbols
        if issubclass(op_type, Definition):
            d = definition(apply, def_symbols[apply.op])
            body.extend(d)
        elif issubclass(op_type, PythonOp):
            assert not python_op_symbols is None, "compiler error: python_op_symbols not passed to _mk_function_definition"
            symbol = python_op_symbols[apply.op]
            d = definition(apply, symbol)
            body.extend(d)
        else:
            op_handler = op_handlers.get(op_type)
            if not op_handler:
                raise ValueError(f"Unknown op {op_type}")
            body.extend(op_handler(apply))

    rval = ast.List(elts=[ load(i) for i in fn.returns ], ctx=ast.Load())
    body.append(ast.Return(value=rval, ctx=ast.Load()))
    return ast.FunctionDef(
        name = name,
        args = args,
        body = body,
        decorator_list=[],
        type_params=[])


def _to_python_class_ast(fs: dict[str, OpenHypergraph], class_name: str = 'Dynamic', python_op_symbols=None):
    # TODO: how do we allow the user to specify additional imports?
    # get each Definition operation as an OpenHypergraph with a private name

    # check all names in fs are not private (i.e., start with _).
    # TODO: clarify these rules for function names somewhere
    for name, f in fs.items():
        _assert_identifier(name)
        if name.startswith("_"):
            raise ValueError(f"function name {name} must not begin with '_'")

    # NOTE: disjoint union of keys because private_def names all start with _
    defs = _extract_definitions(fs)
    def_symbols = { x: symbol for x, (symbol, _) in defs.items() }
    private_defs = { symbol: hyp for _, (symbol, hyp) in defs.items() }
    fs = {**fs, **private_defs}

    # create function definitions for each class
    fn_defs = {}
    for name, f in fs.items():
        fn_defs[name] = _mk_function_definition(f, def_symbols, name=name, python_op_symbols=python_op_symbols)

    mod_ast = _mk_module(class_name, list(fn_defs.values()))
    ast.fix_missing_locations(mod_ast)
    return mod_ast

def to_python_class_ast(fs: dict[str, OpenHypergraph], class_name: str = 'Dynamic'):
    # This function prevents users from calling _to_python_class_ast when
    # ``PythonOp``s are present: these are not added to the AST, so only work
    # correctly with to_python_class (and not to_python_class_ast).
    python_ops = [ x for f in fs.values() for x in f.H.x if isinstance(x, PythonOp) ]
    if len(python_ops) > 0:
        raise ValueError("cannot compile OpenHypergraph with PythonOp to AST")
    return _to_python_class_ast(fs, class_name)

def to_python_class(fs: dict[str, OpenHypergraph], class_name: str = 'Dynamic', return_ast=False):
    filename='<string>'
    # call unsafe _to_python_class_ast because we want to process *with* the ops.
    python_op_symbols = _extract_python_ops(fs)
    mod_ast = _to_python_class_ast(fs, class_name, python_op_symbols=python_op_symbols)
    env: Any = {}
    exec(compile(mod_ast, filename=filename, mode='exec'), env)

    # TODO: make tracebacks work properly for generated member functions.
    Dynamic = env[class_name]

    # incredibly cursed escape hatch: instead of actually turning PythonOps into
    # AST, we just attach them to the class after.
    for x, symbol in python_op_symbols.items():
        setattr(Dynamic, symbol, x)

    if return_ast:
        return Dynamic, mod_ast
    return Dynamic

def to_python_function(f: OpenHypergraph, function_name: str = 'fn', filename='<string>', array_backend=Numpy) -> Callable:
    # compile to a class
    Dynamic = to_python_class({function_name: f}, 'Dynamic')

    # instantiate with numpy array backend and return closure over class
    d = Dynamic(array_backend)
    return (lambda *args: d.fn(*args))
