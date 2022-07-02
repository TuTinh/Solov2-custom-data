from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv


config_file = '/content/drive/MyDrive/RoadCrackDetection/Solov2_7_4/SOLO/configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/content/drive/MyDrive/RoadCrackDetection/Solov2_7_4/SOLO/work_dirs/solov2_light_release_r18_fpn_8gpu_3x/epoch_20.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = '/content/drive/MyDrive/RoadCrackDetection/Solov2_7_4/SOLO/img_test/113.jpg'
result = inference_detector(model, img)

show_result_ins(img, result, ['_background_', 'road_crack'], score_thr=0.25, out_file="demo/road_crack_113.jpg")
# show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo/road_crack_111.jpg")
