"""Reusable training loop (W4, task Z3). From W5 on, every model is trained with it.

Check it with:
    uv run pytest exercises/w04
"""

import torch


def train_one_epoch(model, loader, loss_fn, optimizer):
    """One pass over loader with a step per batch. Returns the mean loss over all examples of the epoch.

    Per batch: zero the gradients, compute the loss, backward, optimizer step. Put the model in training
    mode first (model.train()). The mean is over examples, so weight each batch loss by the batch size:
    the last batch may be smaller.
    """
    # TODO (Z3)
    model.train()
    total_loss = 0.0 # suma strat 
    total_examples = 0 # liczba przykładów 
    for X,y in loader:
        optimizer.zero_grad() # bez wyzerowana gradienty z poprzeniej partii zostaną dodane do gradientów z bieżącej partii
        predictions = model(X) # przetwarza dane wejściowe i zwraca predykcje X @ w + b
        loss = loss_fn(predictions, y)
        loss.backward() #oblicza gradienty (w jakim kierunku zmienić każdy parametr, żeby zmniejszyć stratę)
        optimizer.step() # ktualizacja parametrów modelu za pomocą obliczonych gradientów
        batch_size = X.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size # bo samo loss to tensor, a .item zwraca float. loss to średnia na partię dlatego mnożenie
    return total_loss/total_examples


def evaluate(model, loader, loss_fn):
    """Mean loss over all examples of loader, without changing the model and without gradients.

    Use model.eval() and torch.no_grad(). Weight each batch loss by the batch size.
    """
    # TODO (Z3)
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad(): # wyłącza obliczanie gradientów. w evaulacji gradnienty nie potrzebne, bo gradniety są do uczenia
        for X, y in loader:
            predictions = model(X)
            loss = loss_fn(predictions, y)
            batch_size = X.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss/total_examples

