from catgrad.target.python import to_python_class, to_python_class_ast

from catgrad.combinators import identity
from catgrad.rdops import sgd, mse, SGD, MSE, discard
from catgrad.functor import Bidirectional
from catgrad.rdops import Forget
from catgrad.parameters import factor_parameters

F = Forget()
B = Bidirectional()

# utility function to extra params and compile a model into useful functions.
def compile_model(model, opt, loss):
    params, parametrised = factor_parameters(model)
    P = params.target
    X = model.source
    Y = model.target

    g = (opt(P) @ identity(X)) >> parametrised >> loss(Y)
    Rg = B.adapt(B(g), g.source, g.target)

    # predict_circuit : P × X → Y
    predict_circuit = F(parametrised)

    # only output new params for step, i.e. so
    # step_circuit : P × X × Y → P
    step_circuit = F(Rg >> (discard(Y) @ identity(P) @ discard(X)))

    fns = {
        "predict": predict_circuit,
        "step": step_circuit,
    }
    mod_ast = to_python_class_ast(fns)
    return to_python_class(fns), P, mod_ast
