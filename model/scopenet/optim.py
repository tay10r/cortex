from torch import optim

optimizers = {
    'sgd': lambda params, lr: optim.SGD(params, lr),
    'adam': lambda params, lr: optim.Adam(params, lr),
    'adam_w': lambda params, lr: optim.AdamW(params, lr)
}
