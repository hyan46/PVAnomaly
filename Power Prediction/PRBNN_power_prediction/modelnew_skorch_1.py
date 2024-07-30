
#%%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import logging
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

#%%

from skorch.callbacks import Callback

class ConstraintCallback(Callback):
    def __init__(self, derate_loader,use_constraint = True):
        self.derate_loader = derate_loader
        self.use_constraint = use_constraint
    def on_batch_end(self, net,  training, **kwargs):


        device = net.device
        constraint_loss = 0.0

        
        derate_data = self.derate_loader
        derate_output = net.module_(derate_data)
        de_0=derate_output[:6]
        de_1=derate_output[6:]
        sub=de_0-de_1
        sub[sub<0]=0
        constraint_loss += torch.mean(sub)
        if self.use_constraint:
        # Modifying the loss
            loss = kwargs['loss']
            loss += constraint_loss
            net.history.record_batch('constraint_loss', constraint_loss.item())
        print(f"Constraint Loss = {constraint_loss.item()}")
        
class CustomRegressor(NeuralNetRegressor):
    def __init__(self, *args, derate_loader =None, use_constraint = True,**kwargs):
        super().__init__(*args, **kwargs)
        self.derate_loader = derate_loader
        self.use_constraint = use_constraint
    def train_step_single(self, batch, **fit_params):
        self.module_.train()
        Xi, yi = batch
        yi_pred, mse_loss, kl_loss = self.module_.compute_loss(Xi, yi)
        loss = mse_loss + 0.000018 * kl_loss  # Adjust KL loss contribution

        if self.use_constraint and self.derate_loader is not None:
            constraint_loss = self.compute_constraint_loss()
            loss += constraint_loss

        self.optimizer_.zero_grad()
        loss.backward()
        self.optimizer_.step()


        self.history.record_batch('constraint_loss', constraint_loss.item())
        self.history.record_batch('mse_loss', mse_loss.item())
        self.history.record_batch('kl_loss', kl_loss.item())
        return {'loss': loss, 'mse_loss': mse_loss, 'kl_loss': kl_loss, 'constraint_loss': constraint_loss}
    
    def compute_constraint_loss(self):
        # Example constraint calculation
        constraint_loss = 0
        derate_data = self.derate_loader
        derate_output = self.module_(derate_data)
        de_0=derate_output[:6]
        de_1=derate_output[6:]
        sub=de_0-de_1
        sub[sub<0]=0
        constraint_loss =torch.mean(sub)
            # Calculate your constraint loss
        return constraint_loss
    
    def predict(self, X):
        return super().predict(X)

class LoggingCallback(Callback):
    def on_epoch_end(self, net, **kwargs):
        epoch = net.history[-1]
        mse_loss = np.mean([b['mse_loss'] for b in net.history[-1]['batches'] if 'mse_loss' in b])
        kl_loss = np.mean([b['kl_loss'] for b in net.history[-1]['batches'] if 'kl_loss' in b])
        constraint_loss = np.mean([b['constraint_loss'] for b in net.history[-1]['batches'] if 'constraint_loss' in b])

        print(f"Epoch {epoch['epoch']}: MSE Loss = {mse_loss}, KL Loss = {kl_loss}, Constraint Loss = {constraint_loss}")
            
def setup_model(device, lr, derate_loader=None,use_constraint = True,max_epochs=100,batch_size=32):
    callbacks = []
    # if derate_loader is not None:
    #     callbacks.append(('derate_constraints', ConstraintCallback(derate_loader)))
    callbacks.append(('early_stopping', EarlyStopping(patience=10)))
    # Include any custom callbacks here
    callbacks.append(('logging', LoggingCallback()))
    net = CustomRegressor(
        module=BayesianNet,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=lr,
        device=device,
        derate_loader=derate_loader,
        max_epochs=20,
        batch_size=32,
        callbacks=callbacks
)

    return net


class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01,
          in_features=10, out_features=64)
        self.hid2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.02,#0.02
          in_features=64, out_features=32)
        self.oupt = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01,
          in_features=32, out_features=1)
        self.bnn_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    def forward(self, X):
        X = F.relu(self.hid1(X))
        X = F.relu(self.hid2(X))
        return self.oupt(X)

    def compute_loss(self, X, y):
        prediction = self.forward(X)
        mse_loss = F.mse_loss(prediction, y.view(-1, 1))
        kl_loss = self.bnn_loss(self)
        return prediction, mse_loss, kl_loss


# class BayesianNet(nn.Module):
#     def __init__(self):
#         super(BayesianNet, self).__init__()
#         self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=10, out_features=64)
#         self.hid2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.02, in_features=64, out_features=32)
#         self.oupt = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=32, out_features=1)
#         self.loss_func = torch.nn.MSELoss()
#         self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

#     def forward(self, x):
#         y = F.relu(self.hid1(x))
#         z = F.relu(self.hid2(y))
#         z = self.oupt(z)
#         return z

#     def train_model(self, trainloader, derate_loader=None, epochs=400, device='cuda:0', save_path='./model_checkpoint.pth'):
#         self.to(device)
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)
#         for epoch in range(epochs):
#             self.train()
#             train_loop = tqdm(trainloader, leave=True, position=0,mininterval=100)
#             fit_loss_total = complexity_loss_total = constraint_loss_total = 0

#             for x, y in trainloader:
#                 x, y = x.to(device), y.to(device).view(-1, 1)
#                 # print(x.shape, y.shape,derate_loader.shape)
#                 optimizer.zero_grad()
#                 prediction = self(x).to(device)
#                 loss = self.loss_func(prediction, y) + 0.000018 * self.kl_loss(self)

#                 # Apply constraints if derate_loader is provided
#                 if derate_loader is not None:
#                     derate_output = self(derate_loader.to(device))
#                     de_0=derate_output[:6]
#                     de_1=derate_output[6:]
#                     sub=de_0-de_1
#                     sub[sub<0]=0
#                     con_derate=torch.mean(sub)

#                     loss += con_derate
#                     constraint_loss_total += con_derate.item()

#                 loss.backward()
#                 optimizer.step()
#                 train_loop.set_description(f'Epoch {epoch + 1}/{epochs}')
#                 train_loop.update(1)
#                 train_loop.set_postfix(loss=loss.item())
#                 fit_loss_total += loss.item()

#             # Print and save the model at intervals
#             if (epoch + 1) % 5 == 0:
#                 avg_fit_loss = fit_loss_total / len(trainloader)
#                 avg_constraint_loss = constraint_loss_total / len(trainloader) if derate_loader is not None else 0
#                 logging.info(f'Epoch {epoch + 1}: Fit Loss: {avg_fit_loss}, Constraint Loss: {avg_constraint_loss}')
#                 torch.save(self.state_dict(), os.path.join(save_path,'./model_checkpoint.pth'))
#                 logging.info(f'Model saved to {save_path}')

#     def load_and_test(self, testloader, save_path='./model_checkpoint.pth', n_samples=50, device='cuda:0'):
#         if os.path.exists(save_path):
#             self.load_state_dict(torch.load(os.path.join(save_path,'./model_checkpoint.pth'), map_location=device))
#             self.to(device)
#             self.eval()
#             logging.info(f'Model loaded from {save_path}')
#         else:
#             logging.info('No saved model found at {save_path}. Please check the path.')
#             return None, None

#         test_results = torch.zeros((len(testloader.dataset), n_samples), device=device)
#         test_targets = torch.zeros(len(testloader.dataset), device=device)

#         with torch.no_grad():
#             batch_start = 0
#             for data, target in testloader:
#                 data, target = data.to(device), target.to(device)
#                 # print(data.shape, target.shape)
#                 batch_size = data.size(0)
#                 for i in range(n_samples):
#                     output = self(data).squeeze()
#                     test_results[batch_start:batch_start + batch_size, i] = output
#                 test_targets[batch_start:batch_start+batch_size] = target
#                 batch_start += batch_size

#         return test_results.cpu(), test_targets.cpu()

#     def evaluate(self, predictions, targets):
#         y_true = targets.cpu().numpy()
#         y_pred = predictions.mean(dim=1).cpu().numpy()

#         mse = mean_squared_error(y_true, y_pred)
#         r2 = r2_score(y_true, y_pred)

#         logging.info(f'Mean Squared Error: {mse}')
#         logging.info(f'R-squared: {r2}')
#         return mse, r2
    
#     def visualize_and_save(self, testY, predictions, figure_name, visualization_period=(10, 30)):
#         visualization_period_1, visualization_period_2 = visualization_period
#         epistemic_uncertainty = predictions.std(dim=1)
#         mean_predictions = predictions.mean(dim=1)

#         # Make directory if it does not exist
#         save_directory = os.path.dirname(figure_name)
#         if not os.path.exists(save_directory):
#             os.makedirs(save_directory, exist_ok=True)

#         plt.figure(figsize=(30, 8))
#         plt.title('BNN With Hard Constraint', fontsize=18)

#         # Plot predictions
#         plt.plot(mean_predictions.cpu().numpy(), label='Predicted Mean')
#         plt.fill_between(np.arange(len(testY)), 
#                          (mean_predictions - 3 * epistemic_uncertainty).cpu().numpy(), 
#                          (mean_predictions + 3 * epistemic_uncertainty).cpu().numpy(), 
#                          alpha=0.3, color='blue', label='Epistemic uncertainty')
        
#         # Plot ground truth
#         plt.plot(testY.cpu().numpy(), label='Ground Truth', linestyle='--')

#         # Add legend
#         plt.legend(fontsize=18)
        
#         # Save figure
#         plt.savefig(figure_name)
#         plt.close()
#         print(f'Visualization saved as {figure_name}')