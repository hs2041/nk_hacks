# Import the library
import torch
import torch.nn as nn

# Define the input data as a 3-D tensor
x = torch.tensor([[[0.,1.,2.],
                   [3.,4.,5.],
                   [3.,4.,6.],
                   [6.,7.,8.]]])

print('input_data:', x.shape)

# Create an instance of the nn.Linear class
output_size = 5
linear_layer = nn.Linear(in_features=x.size()[-1],
                         out_features=output_size,
                         bias = True,
                         dtype= torch.float)

# Apply the linear transformation to the input data
y = linear_layer(x)
#print the outputs
print('\noutput:\n',y)

#print the outputs shape
print('\noutput Shape:', y.shape)