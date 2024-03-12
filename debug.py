def evaluate_single(dataset, pred_root, save_txt=None):
    """
    :param dataset: 要测试的数据集，必须为['CHAMELEON', 'CAMO', 'COD10K', 'NC4K', 'CPD1K']之一
    :param pred_root: pred_root中包含了某个测试集下的所有网络预测灰度图，值为[0, 255]；预测图命名与原图像相同。
    :return: None
    """
    assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K', 'CPD1K']
    metricCLS = EvaluationMetricsV2()
    mask_root = getattr(dp, f'test_{dataset}_masks')
    # mask_name_list = sorted(os.listdir(mask_root))
    mask_name_list = sorted(os.listdir(pred_root))
    print(f'{dataset}: ')
    for i, mask_name in tqdm(list(enumerate(mask_name_list))):
        pred_path = os.path.join(pred_root, mask_name)
        mask_path = os.path.join(mask_root, mask_name)
        pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        if pred.shape != mask.shape:
            print(f'shape not equal: {mask_name}, pred: {pred.shape}, mask: {mask.shape}')
            with open(save_txt, 'a') as f:
                f.write(f'shape not equal: {mask_name}, pred: {pred.shape}, mask: {mask.shape}\n')
            pred = torch.from_numpy(pred).unsqueeze(0)
            pred = T.Resize(mask.shape)(pred)
            pred = pred.squeeze(0).numpy()
        assert pred.shape == mask.shape
        # print(pred.shape)   # (811, 1200)
        metricCLS.step(pred=pred, gt=mask)

    metric_dic = metricCLS.get_results()

    sm = metric_dic['sm']
    emMean = metric_dic['emMean']
    emAdp = metric_dic['emAdp']
    wfm = metric_dic['wfm']
    mae = metric_dic['mae']

    print('sm:', sm)
    print('emMean:', emMean)
    print('emAdp:', emAdp)
    print('wfm:', wfm)
    print('mae:', mae)

    if save_txt is not None:
        with open(save_txt, 'a', encoding='utf-8') as f:
            f.write(f'{dataset}:\n')
            f.write(f'SM: {sm}\n')
            f.write(f'meanEM: {emMean}\n')
            f.write(f'adapEM: {emAdp}\n')
            f.write(f'WFM: {wfm}\n')
            f.write(f'MAE: {mae}\n\n')

    return metric_dic