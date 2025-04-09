from tqdm import tqdm
from env import ProjectPaths
import yaml
from .factory import ModelFactory
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainer.load_config()
        self.model_factory = ModelFactory()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = self.config['training']['epochs']


    @staticmethod
    def load_config():
        "Loads the `config.yaml` file into a structured object"

        config_path = ProjectPaths.CONFIG_DIR.value / 'config.yaml'
        
        with open(config_path, 'rb') as f:
            return yaml.safe_load(f)
        
    def save_model(self, model, model_name, save_dir='trained_models/models/'):
        """Save the model state dictionary to a file"""
        save_path = f"{save_dir}{model_name}_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at: {save_path}")
       

    def init_weights(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Apply Xavier initialization for LSTM layers
                    nn.init.xavier_uniform_(param.data)
                else:
                    # Apply He initialization for other layers (if applicable)
                    nn.init.kaiming_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


    def train(self, trainloader: DataLoader, valloader: DataLoader, dataset: str = 'medal', embedding_dim = 100, **kwargs):
        best_model = None
        best_acc = 0

        for model_name in self.config['model_names']:
            print(f'------- {model_name} --------')
            training_hyperparams = self.config['training']['hyperparameters']
            model_hyperparams = self.config['models'][model_name].get('hyperparameters', {})
            model_baseparams = self.config['models'][model_name].get('base_params', {})
            num_classes = self.config['datasets'][dataset]['num_classes']
            create_embedding_layer = self.config['training']['create_embedding_layer']
            embedding_model = None
            if 'embedding_model' in kwargs:
                embedding_model = kwargs['embedding_model']

            # Combine parameters safely
            model_params = {
                **(model_hyperparams if model_hyperparams else {}),
                **(model_baseparams if model_baseparams else {}),
                'num_classes': num_classes,
                'embedding_dim': embedding_dim,
                'create_embedding_layer': create_embedding_layer,
                'embedding_model': embedding_model if embedding_model else None
            }

            print(model_params)

            model: nn.Module = self.model_factory.get_model(model_name, **model_params).to(self.device)
            self.init_weights(model)

            loss_fn = self.config['datasets'][dataset]['loss_function']
            if loss_fn == 'cross_entropy':
                self.criterion = CrossEntropyLoss()  # Use 'none' to compute loss for each token
            
            optimizer_name = self.config['training']['optimizer']

            if optimizer_name == 'adam':
                self.optimizer = Adam(
                    model.parameters(),
                    lr=training_hyperparams['learning_rate'],
                    weight_decay=self.config['training']['weight_decay'])

            results = {}  

            for epoch in range(self.num_epochs):
                total_loss = 0
                correct = 0
                total = 0

                for batch_idx, (inputs, masks, targets) in tqdm(enumerate(trainloader), desc='Training', total=len(trainloader)):
                    inputs, targets, masks = inputs.to(self.device), targets.to(self.device), masks.to(self.device)

                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs, masks)
                    
                    # Compute loss for each token in the sequence
                    loss = self.criterion(outputs, targets)
                    # Loss is already averaged for each sequence, but you can apply masks if needed
                    loss = loss * masks  # Optional if you want to mask padded values
                    loss = loss.sum() / masks.sum()  # Normalize over non-masked values
                    
                    loss.backward()
        
                    # Apply gradient clipping during training
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    self.optimizer.step()

                    total_loss += loss.item()

                    if batch_idx % 15000 == 0:
                        print(f'Loss: {loss.item()}')

                    # Calculate accuracy (ignoring padding tokens)
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == targets).sum().item()

                    total += targets.size(0)

                avg_loss = total_loss / len(trainloader)
                accuracy = 100 * correct / total
                
                print(f'Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

                results[f'epoch_{epoch+1}'] = {'loss': avg_loss, 'accuracy': accuracy}
                self.save_model(model, 'lstm_att_glove')
                print(f'Visualization of {epoch+1}')
                self.plot_results(results)

                
            # Validation logic (as before)
            val_loss, val_acc = self.evaluate(valloader, model, 'Validation')

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model

            results[f'validation'] = {f'loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}'}

        self.save_model(best_model, 'lstm_att_glove')

        return results

    def plot_results(self, results):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # Extract train and validation losses and accuracies from results
        for epoch, metrics in results.items():
            if 'epoch' in epoch:
                train_losses.append(metrics['loss'])
                train_accuracies.append(metrics['accuracy'])
            elif 'validation' in epoch:
                val_losses.append(metrics['loss'])
                val_accuracies.append(metrics['accuracy'])

        # Plot Training and Validation Losses
        plt.figure(figsize=(12, 5))

        # Plot Losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracies
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def evaluate(self, dataloader: DataLoader, model: nn.Module, set: str = 'Validation'):
        model.eval()  
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():  
            for inputs, masks, targets in dataloader:
                inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)

                outputs = model(inputs, masks)
                loss = self.criterion(outputs, targets)
                loss = loss * masks  # Optional if you want to mask padded values
                loss = loss.sum() / masks.sum()

                total_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples

        print(f'{set} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        return avg_loss, accuracy


            

            
    

