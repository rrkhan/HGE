import torch
from torch.utils.data import DataLoader

from lib.geoopt import ManifoldParameter
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.optim.lr_scheduler import MultiStepLR
from models.benchmark_models import HyperbolicCNN, EuclideanCNN



def load_checkpoint(model, optimizer, lr_scheduler, args):
    """ Loads a checkpoint from file-system. """

    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint:
        if checkpoint['args'].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr

            if (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(args.optimizer, checkpoint['args'].optimizer))

    return model, optimizer, lr_scheduler

def select_model(args):
    """ Selects and sets up an available model and returns it. """

    if args.manifold == "lorentz":
        model = HyperbolicCNN(num_classes=args.num_classes,
                              length=args.length,
                              model_dim=args.num_channels,
                              fc_dim=args.embedding_dim,
                              num_layers=args.num_layers,
                              multi_k_model=args.multi_k_model,
                              learnable_k=args.learnable_k,
                              k=args.k
                              )
    elif args.manifold == "euclidean":
        model = EuclideanCNN(num_classes=args.num_classes,
                              length=args.length,
                              model_dim=args.num_channels,
                              fc_dim=args.embedding_dim,
                              num_layers=args.num_layers
                              )
    else:
        raise ValueError("Dataset format not supported.")
    
    return model


def select_optimizer(model, args):
    """ Selects and sets up an available optimizer and returns it. """

    model_parameters = get_param_groups(model, args.manifold_lr, args.manifold_weight_decay)

    if args.optimizer == "RiemannianAdam":
        optimizer = RiemannianAdam(model_parameters, lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, stabilize=1)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model

    lr_scheduler = None
    if args.use_lr_scheduler:
        lr_scheduler = MultiStepLR(
            optimizer, milestones=args.lr_scheduler_milestones, gamma=args.lr_scheduler_gamma
        )
        

    return optimizer, lr_scheduler

def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = ["manifold.k"]

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not isinstance(p, ManifoldParameter)
                and not any(nd in n for nd in k_params)
            ],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and isinstance(p, ManifoldParameter)
            ],
            'lr' : lr_manifold,
            "weight_decay": weight_decay_manifold
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in k_params)
            ], 
            "weight_decay": 0,
            "lr": 1e-4
        }
    ]

    return parameters
