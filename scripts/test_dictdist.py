import torch
import torch.distributed as dist

# Initialize the dictionary on each GPU
dictionary = {'key1': torch.tensor([1, 2, 3]), 'key2': torch.tensor([4, 5, 6])}
device = torch.device('cuda')
dictionary = {key: value.to(device) for key, value in dictionary.items()}

# Collect the dictionary across GPUs
dist.init_process_group(backend='nccl')  # Initialize the distributed environment
world_size = dist.get_world_size()

# Convert the dictionary to a list of tensors
tensor_list = [value for value in dictionary.values()]

# All gather the list of tensors across GPUs
gathered_tensors = [torch.zeros_like(tensor) for tensor in tensor_list]
dist.all_gather(gathered_tensors, tensor_list)

# Convert the gathered tensors back to a dictionary
gathered_dict = {key: gathered_tensors[i] for i, key in enumerate(dictionary.keys())}

# Print the gathered dictionary on each GPU
for i in range(world_size):
    if i == dist.get_rank():
        print(f'Rank {i}: {gathered_dict}')

dist.destroy_process_group()  # Clean up the distributed environment
