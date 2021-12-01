import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Lambda, Compose
import matplotlib.pyplot as plt

import pandas as pd

from training_data_LARGE import LargeOPDSDataset, ToTensor

# import numpy as np

	# # Download training data from open datasets.
	# training_data = datasets.FashionMNIST(
	#     root="data",
	#     train=True,
	#     download=True,
	#     transform=ToTensor(),
	# )

	# # Download test data from open datasets.
	# test_data = datasets.FashionMNIST(
	#     root="data",
	#     train=False,
	#     download=True,
	#     transform=ToTensor(),
	# )
def perform_prediction(dataset_fn, device_list, target_device):
	training_data = LargeOPDSDataset(dataset_fn, train=True, transform=ToTensor())
	test_data = LargeOPDSDataset(dataset_fn, train=False, transform=ToTensor())

	batch_size = 4

	# Create data loaders.
	train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	for sample in test_dataloader:
		X = sample[device_list[0]]
		if len(device_list) > 1:
			for dev in device_list[1:]:
				X = torch.cat((X, sample[dev]), 1)

		# else:
			# X = sample[device_list[0]]
		# X = sample[device_list]
		y = sample[target_device]
		print("Shape of X [N, C, H, W]: ", X.shape)
		print("Shape of y: ", y.shape, y.dtype)
		break

	# exit()

	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print("Using {} device".format(device))

	# Define model
	class NeuralNetwork(nn.Module):
		def __init__(self, n_features):
			super(NeuralNetwork, self).__init__()
			self.flatten = nn.Flatten()
			self.linear_relu_stack = nn.Sequential(
				nn.Linear(n_features, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, 1)
			)

		def forward(self, x):
			x = self.flatten(x)
			logits = self.linear_relu_stack(x)
			return logits

	model = NeuralNetwork(len(device_list)).to(device)
	model = model.float()
	print(model)


	loss_fn = nn.L1Loss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


	def train(dataloader, model, loss_fn, optimizer, device_list, target_device):
		size = len(dataloader.dataset)
		model.train()
		for batch, sample in enumerate(dataloader):
			X = sample[device_list[0]]
			if len(device_list) > 1:
				for dev in device_list[1:]:
					X = torch.cat((X, sample[dev]), 1)

			# X = sample['https://interconnectproject.eu/example/DEKNres2_GI']
			y = sample[target_device]
			X, y = X.to(device), y.to(device)

			# Compute prediction error
			pred = model(X.float())
			# print(y, pred,"\n")
			loss = loss_fn(pred.float(), y.float())


			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch % 100 == 0:
				loss, current = loss.item(), batch * len(X)
				print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
				# print("x", X[0], "y", torch.round(y[0]), "pred", torch.round(pred[0]))
				# print("x", X[0], "y", torch.round(y[0]), "pred", torch.round(pred[0]))


	def test(dataloader, model, loss_fn, device_list, target_device):
		size = len(dataloader.dataset)
		num_batches = len(dataloader)
		model.eval()
		test_loss, correct = 0, 0
		MAE = []
		MAPE= []
		with torch.no_grad():
			for sample in dataloader:
				batch_size = len(sample[device_list[0]])
				X = sample[device_list[0]]
				if len(device_list) > 1:
					for dev in device_list[1:]:
						X = torch.cat((X, sample[dev]), 1)

				# X = sample['https://interconnectproject.eu/example/DEKNres2_GI']
				y = sample[target_device]
				X, y = X.to(device), y.to(device)
				pred = model(X.float())
				test_loss += loss_fn(pred, y).item()
				MAE.append(sum(abs(y - pred))/len(y))
				MAPE.append((sum(abs(y-pred)/abs(y)))/len(y))
				correct += float(sum(abs(y-pred)<0.1))
		MAE = float((sum(MAE)/len(MAE))[0])
		MAPE = float((sum(MAPE)/len(MAPE))[0])*100
		test_loss /= num_batches
		correct /= (size)
		# print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
		print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, MAE: {MAE:>0.3f}, MAPE: {MAPE:>0.1f} Avg loss: {test_loss:>8f} \n")
		return MAPE


	epochs = 15
	model = model.float()
	res = []
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		train(train_dataloader, model, loss_fn, optimizer, device_list, target_device)
		res.append(test(test_dataloader, model, loss_fn, device_list, target_device))
	print("Done!")
	print(res)

if __name__ == '__main__':
	dataset_filename = "res2_subset_table.csv"
	dev_list = ['https://interconnectproject.eu/example/DEKNres2_CP','https://interconnectproject.eu/example/DEKNres2_WM','https://interconnectproject.eu/example/DEKNres2_FR']
	# dataset_filename = "new_table.csv"
	# (including 6_WM) dev_list = ["https://interconnectproject.eu/example/DEKNres3_GE","https://interconnectproject.eu/example/DEKNres4_GE","https://interconnectproject.eu/example/DEKNres3_GI","https://interconnectproject.eu/example/DEKNres3_CP","https://interconnectproject.eu/example/DEKNres6_WM","https://interconnectproject.eu/example/DEKNres3_WM","https://interconnectproject.eu/example/DEKNres3_FR","https://interconnectproject.eu/example/DEKNres5_DW","https://interconnectproject.eu/example/DEKNres5_RF","https://interconnectproject.eu/example/DEKNres3_RF","https://interconnectproject.eu/example/DEKNres2_WM","https://interconnectproject.eu/example/DEKNres6_DW","https://interconnectproject.eu/example/DEKNres1_GI","https://interconnectproject.eu/example/DEKNres6_FR","https://interconnectproject.eu/example/DEKNres4_PV","https://interconnectproject.eu/example/DEKNres4_WM","https://interconnectproject.eu/example/DEKNres1_HP","https://interconnectproject.eu/example/DEKNres1_WM","https://interconnectproject.eu/example/DEKNres1_PV","https://interconnectproject.eu/example/DEKNres3_DW","https://interconnectproject.eu/example/DEKNres4_DW","https://interconnectproject.eu/example/DEKNres6_GE","https://interconnectproject.eu/example/DEKNres3_PV","https://interconnectproject.eu/example/DEKNres1_FR","https://interconnectproject.eu/example/DEKNres5_GI","https://interconnectproject.eu/example/DEKNres6_GI","https://interconnectproject.eu/example/DEKNres2_CP","https://interconnectproject.eu/example/DEKNres6_PV","https://interconnectproject.eu/example/DEKNres2_FR","https://interconnectproject.eu/example/DEKNres2_DW","https://interconnectproject.eu/example/DEKNres6_CP","https://interconnectproject.eu/example/DEKNres5_WM","https://interconnectproject.eu/example/DEKNres4_HP","https://interconnectproject.eu/example/DEKNres1_DW","https://interconnectproject.eu/example/DEKNres4_FR","https://interconnectproject.eu/example/DEKNres4_GI","https://interconnectproject.eu/example/DEKNres4_RF","https://interconnectproject.eu/example/DEKNres4_EV"]
	# dev_list = ["https://interconnectproject.eu/example/DEKNres3_GE","https://interconnectproject.eu/example/DEKNres4_GE","https://interconnectproject.eu/example/DEKNres3_GI","https://interconnectproject.eu/example/DEKNres3_CP","https://interconnectproject.eu/example/DEKNres3_WM","https://interconnectproject.eu/example/DEKNres3_FR","https://interconnectproject.eu/example/DEKNres5_DW","https://interconnectproject.eu/example/DEKNres5_RF","https://interconnectproject.eu/example/DEKNres3_RF","https://interconnectproject.eu/example/DEKNres2_WM","https://interconnectproject.eu/example/DEKNres6_DW","https://interconnectproject.eu/example/DEKNres1_GI","https://interconnectproject.eu/example/DEKNres6_FR","https://interconnectproject.eu/example/DEKNres4_PV","https://interconnectproject.eu/example/DEKNres4_WM","https://interconnectproject.eu/example/DEKNres1_HP","https://interconnectproject.eu/example/DEKNres1_WM","https://interconnectproject.eu/example/DEKNres1_PV","https://interconnectproject.eu/example/DEKNres3_DW","https://interconnectproject.eu/example/DEKNres4_DW","https://interconnectproject.eu/example/DEKNres6_GE","https://interconnectproject.eu/example/DEKNres3_PV","https://interconnectproject.eu/example/DEKNres1_FR","https://interconnectproject.eu/example/DEKNres5_GI","https://interconnectproject.eu/example/DEKNres6_GI","https://interconnectproject.eu/example/DEKNres2_CP","https://interconnectproject.eu/example/DEKNres6_PV","https://interconnectproject.eu/example/DEKNres2_FR","https://interconnectproject.eu/example/DEKNres2_DW","https://interconnectproject.eu/example/DEKNres6_CP","https://interconnectproject.eu/example/DEKNres5_WM","https://interconnectproject.eu/example/DEKNres4_HP","https://interconnectproject.eu/example/DEKNres1_DW","https://interconnectproject.eu/example/DEKNres4_FR","https://interconnectproject.eu/example/DEKNres4_GI","https://interconnectproject.eu/example/DEKNres4_RF","https://interconnectproject.eu/example/DEKNres4_EV"]
	target_dev = 'https://interconnectproject.eu/example/DEKNres2_GI'
	perform_prediction(dataset_filename, dev_list, target_dev)