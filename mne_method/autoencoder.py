from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from preprocess_eeg import load_and_preprocess_eeg_data  # Import the second script

# Load and preprocess EEG data
raw_file_path = 'raw_data/sub-01_task-rsvp_eeg.vhdr'
event_description = 'Event/E  1'
train_data_loader, test_data_loader, n_channels, n_times = load_and_preprocess_eeg_data(raw_file_path, event_description)


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        # Define convolutional layers
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(in_channels=n_channels, out_channels=8, kernel_size=3, stride=2, padding=1),
            # ReLU activation is applied to introduce non-linearity
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            # Batch Normalization helps with training stability and faster convergence
            nn.BatchNorm1d(16),
            # ReLU activation for non-linearity
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=0),
            # ReLU activation for non-linearity
            nn.ReLU(True)
        )
        
        # Adjusted calculation for the output size
        # Assuming the input is padded to a size of 128
        conv_output_size = self._calculate_conv_output_size(128)

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Adjusted input size for the linear layer
        self.encoder_lin = nn.Sequential(
            # Fully connected layer with 32 * conv_output_size input features
            nn.Linear(32 * conv_output_size, 128),
            # ReLU activation for non-linearity
            nn.ReLU(True),
            # Output layer for the encoded space
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.encoder_cnn(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Forward pass through linear layers
        x = self.encoder_lin(x)
        return x

    def _calculate_conv_output_size(self, size):
        # Adjusted size calculation for each convolutional layer
        size = (size + 2 * 1 - 3) // 2 + 1  # First conv layer: Applies convolution, stride 2, padding 1
        size = (size + 2 * 1 - 3) // 2 + 1  # Second conv layer: Applies convolution, stride 2, padding 1
        size = (size + 2 * 0 - 3) // 2 + 1  # Third conv layer: Applies convolution, stride 2, no padding
        return size



class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        # Calculate the output size after the final encoder convolution
        conv_output_size = self._get_conv_output_size(128)  # Use 128 as we padded to this size

        # Define the linear layers for decoding
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),  # Fully connected layer to expand encoded data
            nn.ReLU(True),  # ReLU activation for non-linearity
            nn.Linear(128, 32 * conv_output_size),  # Fully connected layer to prepare for reshaping
            nn.ReLU(True)  # ReLU activation for non-linearity
        )

        # Unflatten the data to prepare for convolutional layers
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, conv_output_size))

        # Define the 1D transpose convolutional layers for upsampling
        # You may need to adjust these layers to ensure the output size is 128
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # Transpose convolution layer 1
            nn.BatchNorm1d(16),  # Batch Normalization for stabilization
            nn.ReLU(True),  # ReLU activation for non-linearity
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),  # Transpose convolution layer 2
            nn.BatchNorm1d(8),  # Batch Normalization for stabilization
            nn.ReLU(True),  # ReLU activation for non-linearity
            nn.ConvTranspose1d(8, n_channels, kernel_size=4, stride=2, padding=1)  # Transpose convolution layer 3
        )
        
    def forward(self, x):
        # Apply linear layers to decode the encoded data
        x = self.decoder_lin(x)
        
        # Reshape the data to match the expected shape for convolutional layers
        x = self.unflatten(x)
        
        # Apply the transpose convolutional layers for upsampling
        x = self.decoder_conv(x)
        
        # Resize the output to match the input size if necessary
        x = F.interpolate(x, size=128, mode='linear', align_corners=True)
        
        # Apply sigmoid activation to ensure values are in the range [0, 1]
        return torch.sigmoid(x)

    def _get_conv_output_size(self, size):
        # Adjusted size calculation for each convolutional layer
        size = (size + 2 * 1 - 4) // 2 + 1  # First transpose conv layer
        size = (size + 2 * 1 - 4) // 2 + 1  # Second transpose conv layer
        size = (size + 2 * 1 - 4) // 2 + 1  # Third transpose conv layer
        return size


#change encoded_space_dim to change the size of the encoded space aka the emebdding size for next models
def initialize_autoencoder(encoded_space_dim, lr):
    # Define the loss function and optimizer
    loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss function measures the reconstruction error

    # Initialize the encoder and decoder
    encoder = Encoder(encoded_space_dim=encoded_space_dim)  # Create an instance of the Encoder class with the specified dimension
    decoder = Decoder(encoded_space_dim=encoded_space_dim)  # Create an instance of the Decoder class with the same dimension

    # Define the parameters to optimize, including those from the encoder and decoder
    params_to_optimize = [
        {'params': encoder.parameters()},  # Parameters from the encoder
        {'params': decoder.parameters()}  # Parameters from the decoder
    ]

    # Initialize the optimizer with Adam optimizer, specifying learning rate and weight decay
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available and select the appropriate device (GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    return encoder, decoder, optimizer, device, loss_fn



def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set both the encoder and decoder to training mode
    encoder.train()
    decoder.train()
    
    # Initialize a list to store the training loss for each batch
    train_loss = []
    
    # Iterate through the data loader (usually containing one batch)
    for eeg_batch in dataloader:
        # Extract the EEG data batch and move it to the specified device (GPU or CPU)
        eeg_batch = eeg_batch[0].to(device)
        
        # Forward pass: Encode the EEG data and then decode it
        encoded_data = encoder(eeg_batch)
        decoded_data = decoder(encoded_data)
        
        # Calculate the loss between the decoded data and the original EEG data
        loss = loss_fn(decoded_data, eeg_batch)
        
        # Zero the gradients in the optimizer
        optimizer.zero_grad()
        
        # Backpropagate the loss to compute gradients
        loss.backward()
        
        # Update the model's parameters using the optimizer
        optimizer.step()
        
        # Print the training loss for the current batch
        #print('\t partial train loss (single batch): %f' % (loss.data))
        
        # Append the loss value to the list of training losses
        train_loss.append(loss.detach().cpu().numpy())

    # Calculate the mean training loss for the entire epoch
    return np.mean(train_loss)


if __name__ == "__main__":
    num_epochs = 3  # Number of epochs
    encoder, decoder, optimizer, device,loss_fn = initialize_autoencoder(encoded_space_dim=4, lr=0.001)
    for epoch in range(num_epochs):
        train_loss = train_epoch_den(encoder, decoder, device, train_data_loader, loss_fn, optimizer)
        print(f'Epoch {epoch}: Train Loss: {train_loss}')