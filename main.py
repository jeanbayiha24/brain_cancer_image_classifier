import torch
from utils import prep
import argparse
from models.cnn import CNN, get_tensorflow_model 
from models.train import Trainer, TFTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="CNN model training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="weight decay")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode of operation: 'train' or 'eval' (default: train)")
    parser.add_argument('--model', type=str, choices=['pytorch', 'tf'], default='tf', help="Model to train: 'pytorch' or 'tf' (default: tf)")
    parser.add_argument('--cuda', action='store_true', help="Use GPU if available")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    if args.model == 'tf':
        train_dataloader, test_dataloader = prep.get_tf_data()
        model = get_tensorflow_model()
        trainer = TFTrainer(model, train_dataloader, test_dataloader, args.lr, args.epochs)
    
        if args.mode == 'eval':
            model.load_weights("jean_bayiha_model.weights.h5")
    else:
        train_dataloader, test_dataloader = prep.get_data()
        model = CNN().to(device)
        trainer = Trainer(model, train_dataloader, test_dataloader, args.lr, args.wd, args.epochs, device)
        
        if args.mode == 'eval':
            model.load_state_dict(torch.load("jean_bayiha_model.torch"))
    
    if args.mode == 'train':
        trainer.train(True, True)

    trainer.evaluate()

if __name__ == '__main__':
    main()