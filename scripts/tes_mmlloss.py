# import torch
# import torch.nn.functional as F

# # Define the text meta
# text_meta = [['motorcycle', 'man', 'back', '<PAD>', '<PAD>'],
#              ['cake', 'large white sheet cake', 'sheet', 'woman', '<PAD>'],
#              ['yak', 'flowered umbrella', 'child', 'umbrella', '<PAD>'],
#              ['monitor', 'young boy', 'computer', 'computer monitor', 'boy'],
#              ['desk', 'group', 'computers', 'people', '<PAD>']]

# # Generate a tensor text_meta*text_meta[1] with the positive labels
# pos_labels = F.one_hot(torch.tensor([0, 1, 2, 3, 4]), num_classes=5).float() * 0.2
# print(pos_labels)
# import torch

# pos_labels_batch_img = torch.tensor([[[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
#                                      [[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
#                                      [[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
#                                      [[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
#                                      [[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]]])

# new_tensor = torch.cat((pos_labels_batch_img.flip(1), torch.zeros_like(pos_labels_batch_img)), dim=1)

# print("nt",new_tensor)



# # # Define the positive labels batch image tensor
# # pos_labels_batch_img = torch.tensor([[[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
# #                                      [[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
# #                                      [[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
# #                                      [[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]],
# #                                      [[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]]])

# # # Get the unique positive words
# # positive_words = sorted(list(set([word for batch in text_meta for word in batch if word != '<PAD>'])))

# # # Initialize the tensor
# # tensor_size = (len(text_meta), len(positive_words))
# # tensor = torch.zeros(tensor_size)

# # # Populate the tensor with positive word weights
# # for i, batch in enumerate(text_meta):
# #     for j, word in enumerate(positive_words):
# #         if word in batch:
# #             tensor[i, j] = pos_labels_batch_img[i, 0, positive_words.index(word)]

# # # Print the tensor
# # print(tensor)
import torch

tensor1 = torch.tensor([1, 1, 0, 0, 0, 0])
tensor2 = ['a', 'b', 'c', 'd', 'a', 'a']

result = []
for value, string in zip(tensor1, tensor2):
    if value == 1:
        result.append(1 if string in ['a', 'b'] else 0)
    else:
        result.append(0)

result_tensor = torch.tensor(result)
print(result_tensor)
