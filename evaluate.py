import os
import cv2
from tqdm import tqdm
from config import Config
from utils.metrics import EvaluationMetrics


def evaluate(pred_path, dataset):
	global cfg
	pred_root = os.path.join(pred_path, dataset)
	metric = EvaluationMetrics()
	mask_root = getattr(cfg.dp, f'test_{dataset}_masks')
	mask_name_list = sorted(os.listdir(pred_root))

	for i, mask_name in tqdm(list(enumerate(mask_name_list))):
		pred_path = os.path.join(pred_root, mask_name)
		mask_path = os.path.join(mask_root, mask_name)
		pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
		mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
		assert pred.shape == mask.shape
		metric.step(pred=pred, gt=mask)

	metric_dic = metric.get_results()
	return metric_dic


if __name__ == '__main__':
	pred_path = 'prediction_maps-tinyneta'

	cfg = Config()

	datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
	for dataset in datasets:
		metric_dic = evaluate(pred_path, dataset)

		sm = metric_dic['sm']
		emMean = metric_dic['emMean']
		emAdp = metric_dic['emAdp']
		wfm = metric_dic['wfm']
		mae = metric_dic['mae']

		print(f'{dataset}:')
		print('sm:', sm)
		print('emMean:', emMean)
		print('emAdp:', emAdp)
		print('wfm:', wfm)
		print('mae:', mae)
