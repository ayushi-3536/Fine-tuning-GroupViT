import torch

string_list = ['Hello', 'World']
#string_tensor = [torch.tensor(string_list) for string in string_list]
device_string_list = [torch.tensor(ord(char), device='cuda') for string in string_list for char in string]
print(device_string_list)
gathered_tensors = [torch.empty_like(tensor) for tensor in device_string_list]
torch.distributed.all_gather(gathered_tensors, device_string_list)
# Convert tensors back to strings
collected_string_list = [''.join(chr(tensor.item()) for tensor in gathered_tensors)]

# Print the collected list of strings
for string in collected_string_list:
    print(string)
#print(string_tensor)

