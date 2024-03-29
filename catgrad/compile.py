from catgrad.target.python import to_python_class, to_python_class_ast

from catgrad.combinators import identity
from catgrad.bidirectional.operation import sgd, mse, SGD, MSE, discard
from catgrad.bidirectional.functor import Forget, Bidirectional
from catgrad.special.parameter import factor_parameters

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

    rev = rdiff(parametrised)
    rev_p = rev >> (identity(P) @ discard(X))
    fns = {
        "predict": predict_circuit,
        "rev": F(rev), # gradients
        "rev_p": F(rev_p), # gradients w.r.t. parameters only
        "step": step_circuit,
    }
    Dynamic, mod_ast = to_python_class(fns, return_ast=True)
    return Dynamic, P, mod_ast

def rdiff(f):
    """ Take the reverse derivative of a map built from RDOps """
    f_Rf = B.adapt(B(f), f.source, f.target)
    Rf = f_Rf >> (discard(f.target) @ identity(f.source))
    return Rf
