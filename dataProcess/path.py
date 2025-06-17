import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

os.environ['original_data_local_path'] = "E:/HSX_projects/pancreatic_cancer/pancreatic ct time series 658"
os.environ['raw_data_base'] = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas"
os.environ['divided_data_base'] = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/divided_data"
os.environ['seg_data_base'] = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/seg_data_2D"
os.environ['lesion_data_base'] = "/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/lesion_data_2D"
os.environ['RESULTS_FOLDER'] = "/data/hsx/project_pc_segmentation/nnUNet/nnUNetFrame/DATASET/nnUNet_trained_models"

local_path = os.environ['original_data_local_path']
base = os.environ['raw_data_base'] if "raw_data_base" in os.environ.keys() else None
divided_data_dir = os.environ['divided_data_base'] if "divided_data_base" in os.environ.keys() else None
seg_data_dir = os.environ['seg_data_base'] if "seg_data_base" in os.environ.keys() else None
lesion_data_dir = os.environ['lesion_data_base'] if "lesion_data_base" in os.environ.keys() else None
network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER']) if "RESULTS_FOLDER" in os.environ.keys() else None

# 2D 路径 其中2D指进行2D预处理
base_2D = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas/preprocessed/preprocess_plans_2D'

if base_2D is not None:
    resampled_data_2D = join(base_2D, "stage_resample_xy")
    window_data_2D = join(base_2D, "stage_window")
    nnu_window_data_2D = join(base_2D, "stage_nnu_window")
    seg_data_2D = join(base_2D, "stage_seg") # 存储用于投入分割模型的预处理结果
    maybe_mkdir_p(resampled_data_2D)
    maybe_mkdir_p(window_data_2D)
    maybe_mkdir_p(nnu_window_data_2D)
    maybe_mkdir_p(seg_data_2D)

# base数据夹组装原始数据--裁剪数据--重采样&标准化数据
if base is not None:
    raw_data = join(base, "raw_data")
    cropped_data = join(base, "cropped_data")
    preprocessing_output_dir = join(base, "preprocessed")
    maybe_mkdir_p(raw_data)
    maybe_mkdir_p(cropped_data)
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("未定义数据base文件夹")

# divided_data_dir数据夹分散a_set d_set v_set
if divided_data_dir is not None:
    a_set = join(divided_data_dir, "a_set")
    v_set = join(divided_data_dir, "v_set")
    d_set = join(divided_data_dir, "d_set")
    maybe_mkdir_p(a_set)
    maybe_mkdir_p(v_set)
    maybe_mkdir_p(d_set)
else:
    print("未定义数据divided_dat_dir文件夹")

# seg_data_dir 用于存储需投入使用的分割slice图像
