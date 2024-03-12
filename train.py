import torch
import random
import numpy as np
import torch.nn.functional as F
from utils.dataloader_freq import TrainDataset
from utils.LRScheduler import CosineDecay
from config import Config
from tqdm import tqdm


def structure_loss(logits, mask):
	"""
    loss function (ref: F3Net-AAAI-2020)

    pred: logits without activation
    mask: binary mask {0, 1}
    """
	weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
	wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='mean')
	wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

	pred = torch.sigmoid(logits)
	inter = ((pred * mask) * weit).sum(dim=(2, 3))
	union = ((pred + mask) * weit).sum(dim=(2, 3))
	wiou = 1 - (inter + 1) / (union - inter + 1)
	return (wbce + wiou).mean()


def train():
	global model, train_datald, optimizer, cfg, scheduler
	for epoch in range(cfg.epochs):
		model.train()

		loss_iter = []
		for img, mask, high, low in tqdm(train_datald):
			optimizer.zero_grad()

			img = img.to(cfg.device)
			mask = mask.to(cfg.device)
			high = high.to(cfg.device)
			low = low.to(cfg.device)

			out1, out2, out3, out4 = model(img, high, low)
			loss1 = structure_loss(out1, mask)
			loss2 = structure_loss(out2, mask)
			loss3 = structure_loss(out3, mask)
			loss4 = structure_loss(out4, mask)
			loss = loss1 + loss2 + loss3 + loss4
			loss.backward()
			optimizer.step()
			loss_iter.append(loss.item())

		print(f'Epoch: {epoch + 1}, LR: {np.round(scheduler.get_lr(), 8)}, Loss: {np.round(np.mean(loss_iter), 8)}')
		scheduler.step()

		if (epoch+1) % 5 == 0 or epoch == cfg.epochs-1:
			torch.save(model.state_dict(),  f'save_pth/epoch_{epoch+1}.pth')


if __name__ == '__main__':
	seed = 123456
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False

	cfg = Config()

	from Model.FINet import FINet
	model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64)).to(cfg.device)
	# model = FINet(backbone='tinynet-a', channels=(8,24,32,64)).to(cfg.device)

	train_dataset = TrainDataset(image_root=cfg.dp.train_imgs, gt_root=cfg.dp.train_masks, trainsize=cfg.trainsize,
	                             edge_root=None)
	train_datald = torch.utils.data.DataLoader(dataset=train_dataset,
	                                           batch_size=cfg.batch_size,
	                                           shuffle=True,
	                                           num_workers=cfg.num_workers,
	                                           pin_memory=True)


	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
	scheduler = CosineDecay(optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs)

	train()
