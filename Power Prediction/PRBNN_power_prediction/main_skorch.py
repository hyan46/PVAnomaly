import torch
from data_utils_skorch import load_data, preprocess_data, normalize_data, split_data,derate_dataloader
from modelnew_skorch_1 import BayesianNet, ConstraintCallback, setup_model
import argparse
import os
import logging
from skorch import NeuralNetRegressor
from sklearn.metrics import mean_squared_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Bayesian Neural Network with optional constraints.")
    parser.add_argument("--device", default="cuda:0", type=str, help="Computation device to use ('cuda:X', 'cpu')")
    parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate for optimizer")
    parser.add_argument("--data_path", default="Pacweather_1.pkl", type=str, help="File path for the input data")
    parser.add_argument("--epochs", default=2, type=int, help="Number of epochs to train the model")
    parser.add_argument("--save_path", default="./exps/result_skorch", type=str, help="Path to save the trained model")
    parser.add_argument("--use_constraints", action="store_true", help="Use constraints during training")
    return parser.parse_args()

def setup_logging(save_path):
    logging.basicConfig(filename=f'{save_path}/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler()) 


def save_args(args, filename="experiment_config.txt"):
    with open(filename, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Configuration saved to {filename}")

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main():
    args = parse_args()
    save_directory = ensure_directory(args.save_path)
    setup_logging(args.save_path)
    # Save parsed arguments to a file within the save directory
    save_args(args, filename=os.path.join(save_directory, "experiment_config.txt"))
    device = args.device if torch.cuda.is_available() else "cpu"
    print(device)
    # Load and preprocess data
    raw_data = load_data(file_path=args.data_path)
    processed_data = preprocess_data(raw_data)
    X,Y = normalize_data(processed_data)
    Y = Y.reshape(-1,1)
    x_train, x_test, y_train, y_test = split_data(X,Y,train_ratio=0.7)
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)    
    if args.use_constraints:
        derate_loader = derate_dataloader(device = device)
    else:
        derate_loader = None
    # derate_loader = derate_dataloader(device = device)
    
    # Initialize model with learning rate and epochs from args
    model = setup_model(lr=args.lr, device=device,derate_loader=derate_loader,use_constraint= args.use_constraints,max_epochs=args.epochs,batch_size=32  ) 
    print(derate_loader.device)
    print(model.device)
    print(x_train.device, y_train.device)
    # Optionally use constraints
    
    
    # Train model
    model.fit(x_train, y_train)
    torch.save(model.module_.state_dict(), os.path.join(args.save_path, 'final_model.pt'))

    # Evaluate model
    y_pred = model.predict(x_test)
    y_pred = y_pred.cpu().numpy()
    y_test = y_test.cpu().numpy()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Test MSE: {mse}, R^2: {r2}")


if __name__ == "__main__":
    
    main()
