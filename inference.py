import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from Model.FINet import FINet
from config import Config
from utils.dataloader_freq import TestDataset


def inference(datasets):
	global model, cfg
	model.eval()
	for dataset in datasets:
		assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
		save_path = os.path.join('prediction_maps-tinyneta', dataset)
		os.makedirs(save_path, exist_ok=True)

		test_dataset = TestDataset(image_root=getattr(cfg.dp, f'test_{dataset}_imgs'),
		                           gt_root=getattr(cfg.dp, f'test_{dataset}_masks'),
		                           testsize=cfg.trainsize)

		# image, gt, gt_origin, name, high, low
		for img, _, gt, name, high, low in tqdm(test_dataset):
			img = img.unsqueeze(0).cuda()
			high = high.unsqueeze(0).cuda()
			low = low.unsqueeze(0).cuda()
			out1 = model(img, high, low)[0]
			out1 = F.interpolate(out1, size=gt.shape[1:], mode='bilinear', align_corners=True)
			out1 = torch.sigmoid(out1) * 255
			out1 = out1.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
			# save preds
			cv2.imwrite(os.path.join(save_path, name), out1)


if __name__ == '__main__':
	pth_path = 'FINet.pth'
	# pth_path = 'FINet-TinyNetA.pth'

	cfg = Config()
	model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)
	# model = FINet(backbone='tinynet-a', channels=(8, 24, 32, 64)).to(cfg.device)
	model.load_state_dict(torch.load(pth_path))

	datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
	inference(datasets)
