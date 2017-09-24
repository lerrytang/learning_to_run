import numpy as np
import chainer.functions as F


def ordered_params(chain):
    namedparams = sorted(chain.namedparams(), key=lambda x: x[0])
    return [x[1] for x in namedparams]


def get_flat_params(chain):
    xp = chain.xp
    params = ordered_params(chain)
    if len(params) > 0:
        return xp.concatenate([xp.ravel(param.data) for param in params])
    else:
        return xp.zeros((0,), dtype=xp.float32)


def get_flat_grad(chain):
    xp = chain.xp
    params = ordered_params(chain)
    if len(params) > 0:
        return xp.concatenate([xp.ravel(param.grad) for param in params])
    else:
        return xp.zeros((0,), dtype=xp.float32)


def set_flat_params(chain, flat_params):
    offset = 0
    for param in ordered_params(chain):
        param.data[:] = flat_params[offset:offset +
                                           param.data.size].reshape(param.data.shape)
        offset += param.data.size


def set_flat_grad(chain, flat_grad):
    offset = 0
    for param in ordered_params(chain):
        param.grad[:] = flat_grad[offset:offset +
                                         param.grad.size].reshape(param.grad.shape)
        offset += param.grad.size


def cg(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Demmel p 312. Approximately solve x = A^{-1}b, or Ax = b, where we only have access to f: x -> Ax
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x


