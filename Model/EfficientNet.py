import torch
from torch import nn
from torchvision import models


class EfficientNet_B0(nn.Module):
	def __init__(self):
		super(EfficientNet_B0, self).__init__()

		model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

		self.layer1 = model.features[0]
		# main layers: layer2 ~ layer8
		self.layer2 = model.features[1]
		self.layer3 = model.features[2]  # sup
		self.layer4 = model.features[3]
		self.layer5 = model.features[4]
		self.layer6 = model.features[5]  # sup
		self.layer7 = model.features[6]
		self.layer8 = model.features[7]  # sup
		# last conv 1×1
		# self.layer9 = model.features[8]

	def forward(self, x):
		# torch.Size([1, 32, 112, 112])
		out1 = self.layer1(x)
		# torch.Size([1, 16, 112, 112])
		out2 = self.layer2(out1)

		# torch.Size([1, 24, 56, 56])
		out3 = self.layer3(out2)

		# torch.Size([1, 40, 28, 28])
		out4 = self.layer4(out3)

		# torch.Size([1, 80, 14, 14])
		out5 = self.layer5(out4)
		# torch.Size([1, 112, 14, 14])
		out6 = self.layer6(out5)

		# torch.Size([1, 192, 7, 7])
		out7 = self.layer7(out6)
		# torch.Size([1, 320, 7, 7])
		out8 = self.layer8(out7)
		# torch.Size([1, 1280, 7, 7])
		# out9 = self.layer9(out8)

		return out2, out3, out4, out6, out8

	@staticmethod
	def get_stage_channels():
		return [16, 24, 40, 112, 320]
