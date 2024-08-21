import torch
import torchvision
import genomic_benchmarks
import torch.nn as nn
import torch.nn.functional as F
import configargparse
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
## Training and Finetuning Block
import torch.optim as optim
from utils.initialize import select_model, select_optimizer, load_checkpoint
from utils.data_utils import select_dataset
from sklearn.metrics import matthews_corrcoef


def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='Genomic classification task training', add_help=True)

    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str,
                        help="Path to config file.")

    # Output settings
    parser.add_argument('--run_name', default="test", type=str,
                        help="Name of the experiment.")
    parser.add_argument('--benchmark', default=None, type=str,
                        choices=['TEB', 'GUE'],
                        help="Name of the benchmark used.")
    parser.add_argument('--dataset_name', default=None, type=str,
                        help="Name of dataset.")
    parser.add_argument('--output_dir', default=None, type=str,
                    help="Path for output files.")
    parser.add_argument('--data_path', default=None, type=str,
                    help="Path to the datasets.")
    
    # Experiment settings
    parser.add_argument('--seed', default=1234, type=int,
                        help="Set seed for deterministic training.")
    parser.add_argument('--num_classes', default=2, type=int,
                        help="Number of classes in the data.")
    parser.add_argument('--num_epochs', default=100, type=int,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', default=100, type=int,
                        help="Training batch size.")
    parser.add_argument('--length', default=200, type=int,
                        help="Maximum length of sequences used for the classification task.")
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help="Path to model checkpoint (weights, optimizer, epoch).")    
    parser.add_argument('--lr', default=0.00001, type=float,
                        help="Training learning rate.")
    parser.add_argument('--weight_decay', default=0.1, type=float,
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--manifold_lr', default=2e-2, type=float,
                        help="Learning rate for manifold parameters.")
    parser.add_argument('--manifold_weight_decay', default=5e-4, type=float,
                        help="Weight decay for manifold parameters")
    parser.add_argument('--optimizer', default="RiemannianAdam", type=str,
                        choices=["RiemannianAdam", "RiemannianSGD", "Adam", "SGD"],
                        help="Optimizer for training.")
    parser.add_argument('--use_lr_scheduler', default=True,
                        help="If learning rate should be reduced after step epochs using a LR scheduler.")
    parser.add_argument('--lr_scheduler_milestones', default=[60, 85], type=int, nargs="+",
                        help="Milestones of LR scheduler.")
    parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float,
                        help="Gamma parameter of LR scheduler.")    
    parser.add_argument('--device', default="cuda", type=str)
    
    # Model settings
    parser.add_argument('--num_channels', default=32, type=int,
                        help="Number of channels in conv blocks.")
    parser.add_argument('--num_layers', default=3, type=int,
                        help="Number of conv blocks.")
    parser.add_argument('--embedding_dim', default=528, type=int,
                        help="Dimensionality of classification embedding space (could be expanded by ResNet)")
    parser.add_argument('--manifold', default='lorentz', type=str, choices=["euclidean", "lorentz"],
                        help="Select CNN model manifold.")

    # Hyperbolic geometry settings
    parser.add_argument('--learnable_k', default=True, type=bool,
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--multi_k_model', default=False, type=bool,
                        help="Set hyperbolic model to have multiple manifolds or a single manifold")  

    args = parser.parse_args()
    return args


def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10):
    """Training loop."""
    model.train()
    correct = 0
    targets = []
    outputs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.squeeze())
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        targets.append(target.view(output.shape[0]).cpu())
        outputs.append(pred.view(output.shape[0]).cpu())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('\nTrain set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)

    mcc = matthews_corrcoef(targets, outputs)

    print('MCC:', mcc)
    
def evaluate(model, device, data_loader, loss_fn):
    """Validation loop."""
    model.eval()
    val_loss = 0
    correct = 0
    targets = []
    outputs = []
    
    with torch.no_grad():
        
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target.squeeze()).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            targets.append(target.view(output.shape[0]).cpu())
            outputs.append(pred.view(output.shape[0]).cpu())

    val_loss /= len(data_loader.dataset)

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)

    mcc = matthews_corrcoef(targets, outputs)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    print('MCC:', mcc)
    return mcc




def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("Running experiment: " + args.run_name)

    print("Arguments:")
    print(args)

    print("Loading dataset...")
    train_loader, val_loader, test_loader = select_dataset(args)

    print("Creating model...")
    model = select_model(args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, args)
    loss_fn = nn.CrossEntropyLoss()

    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model, optimizer, lr_scheduler = load_checkpoint(model, optimizer, lr_scheduler, args)

    print("Training...")
    best_mcc = 0.0
    best_epoch = 0

    for epoch in range(args.num_epochs):
        
        train(model, device, train_loader, optimizer, epoch, loss_fn)
        mcc = evaluate(model, device, val_loader, loss_fn)

        if lr_scheduler is not None:
            if (epoch + 1) == args.lr_scheduler_milestones[0]:  # skip the first drop for some Parameters
                optimizer.param_groups[1]['lr'] *= (1 / args.lr_scheduler_gamma) # Manifold params
                print("Skipped lr drop for manifold parameters")            
            lr_scheduler.step()

        if (mcc > best_mcc):
            best_mcc = mcc
            if args.output_dir is not None:
                save_path = args.output_dir + "/best_" + args.run_name + ".pth"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'epoch': epoch,
                    'args': args,
                }, save_path)        

    print("-----------------\nTraining finished\n-----------------")
    print("Best epoch = {}, with MCC={:.4f}".format(best_epoch, best_mcc))

    if args.output_dir is not None:
        save_path = args.output_dir + "/final_" + args.run_name + ".pth"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epoch': epoch,
            'args': args,
        }, save_path)
        print("Model saved to " + save_path)
    else:
        print("Model not saved.")


    print("Testing best model...")
    if args.output_dir is not None:
        save_path = args.output_dir + "/best_" + args.run_name + ".pth"
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)        
        mcc = evaluate(model, device, test_loader, loss_fn)
        print("Performance on test dataset: MCC={:.4f}".format(mcc))
    else:
        print("Best model not saved, because no output_dir given.")



# ----------------------------------
if __name__ == '__main__':
    args = getArguments()
    main(args)