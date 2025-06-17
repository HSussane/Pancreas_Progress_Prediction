from dataProcess.dataset_analyzer import DatasetAnalyzer
from dataProcess.sanity_check import verify_dataset_integrity, verify_same_geometry
from dataProcess.utils import get_data_json_file
import numpy as np
from os.path import join
from dataProcess.utils import get_case_identifier, load_crop_save, create_lists_from_splitted_dataset, get_case_identifier_from_npz, \
    do_case_resample_normalize, plan_resample
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from path import raw_data, cropped_data, preprocessing_output_dir


def run_preprocessing(target_spacing, input_folder_with_cropped_npz, output_folder, data_identifier, transpose_forward,
                      num_threads=8, force_separate_z=None):
    """
    这是预处理类跑预处理的run函数 分进程调用处理函数
    :param target_spacing: list of lists [[1.25, 1.25, 5]] --- []只有一个
    :param input_folder_with_cropped_npz: dim: c, x, y, z | npz_file['data'] np.savez_compressed(fname.npz, data=arr)
    :param output_folder: nnUNetData_plans_v2.1_stage0
    :param num_threads: 8
    :param force_separate_z: None 这里默认是none才可以保证后续自适应调整是否单独重采样z
    """
    print("Initializing to run preprocessing")
    print("npz folder:", input_folder_with_cropped_npz)
    print("output_folder:", output_folder)
    list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
    maybe_mkdir_p(output_folder)
    all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']
    # 这里stage0定义数据预处理 stage1定义网络训练推理中的架构、参数、策略 所以可以简化代码
    all_args = []
    output_folder_stage = os.path.join(output_folder, data_identifier)
    maybe_mkdir_p(output_folder_stage)
    for j, case in enumerate(list_of_cropped_npz_files):
        case_identifier = get_case_identifier_from_npz(case)
        args = target_spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, transpose_forward, force_separate_z, all_classes
        all_args.append(args)
    p = Pool(num_threads)
    p.starmap(do_case_resample_normalize, all_args)
    p.close()
    p.join()

def prepare_preprocess(num_threads):
    if os.path.isdir(join(preprocessing_output_dir, "gt_segmentations")):
        shutil.rmtree(join(preprocessing_output_dir, "gt_segmentations"))
    shutil.copytree(join(cropped_data, "gt_segmentations"), join(preprocessing_output_dir, "gt_segmentations"))
    plans = load_pickle(join(preprocessing_output_dir, "preprocess_plans_3D.pkl")) # 这是保存的plans 预处理关键信息
    target_spacing = np.array(plans['target_spacings']) # [i["current_spacing"] for i in self.plans_per_stage.values()]
    run_preprocessing(target_spacing, cropped_data, preprocessing_output_dir, plans['data_identifier'], plans['transpose_forward'], num_threads)

def run_cropping(list_of_files, output_folder, overwrite_existing=False, num_threads=8):
    """
    所有文件的list传入该函数，然后逐一分散到多进程处理每一个case
    复制mask的原始数据到gt_segmentation
    :param list_of_files: list of list of files [[PATIENTID_TIMESTEP.nii.gz], [PATIENTID_TIMESTEP.nii.gz]]
    :param overwrite_existing: 默认不重写文件夹数据
    :param output_folder: cropped_data
    :return:
    """
    output_folder_gt = os.path.join(output_folder, "gt_segmentations")
    maybe_mkdir_p(output_folder_gt)
    for j, case in enumerate(list_of_files):
        if case[-1] is not None:
            shutil.copy(case[-1], output_folder_gt) # shutil移动具体文件到指定已创建目录或文件

    list_of_args = []
    for j, case in enumerate(list_of_files):
        case_identifier = get_case_identifier(case)
        list_of_args.append((case, case_identifier, overwrite_existing))

    p = Pool(num_threads)
    p.starmap(load_crop_save, list_of_args) # 瞎了眼的 这里多线程调用了函数load_crop_save
    p.close()
    p.join()

def prepare_cropping(override=False, num_threads=8):
    maybe_mkdir_p(cropped_data)
    if override and isdir(cropped_data): # 设置了重写就会清空
        shutil.rmtree(cropped_data)
        maybe_mkdir_p(cropped_data)
    lists, _ = create_lists_from_splitted_dataset(raw_data) # list：每一组CT-mask的配对列表
    run_cropping(lists, cropped_data, overwrite_existing=override, num_threads=num_threads)
    shutil.copy(join(raw_data, "dataset.json"), cropped_data)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_data_json", required=False, default=False)
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    args = parser.parse_args()

    # raw_sec_data = '/data/hsx/project_pc_multimodality_addsize/Task_Pancreas_sec/raw_data'
    if args.generate_data_json:
        get_data_json_file(raw_data)

    if args.verify_dataset_integrity:
        verify_dataset_integrity(raw_data)
    # 经过上面检测 最后还有 10675323_6_a 10675323_6_v两个direction不一样 所以单独调整
    # case_name = '10675323_6_v'
    # images_itk = sitk.ReadImage(join(raw_data,'images',case_name+'.nii.gz'))
    # mask_itk = sitk.ReadImage(join(raw_data,'masks',case_name+'.nii.gz'))
    # verify_same_geometry(images_itk, mask_itk, case_name)

    # step.1 crop数据保存信息为npz已经实现 呜呜呜
    # prepare_cropping(True, args.tf)
    # step.2 统计cropped_dataset的数据集信息，以便后续重采样标准化读取和计算
    # dataset_analyzer = DatasetAnalyzer(cropped_data, overwrite=True, num_processes=args.tf)  # this class creates the fingerprint
    # _ = dataset_analyzer.analyze_dataset() # 如果图像是CT 默认就是true
    # maybe_mkdir_p(preprocessing_output_dir)
    # shutil.copy(join(cropped_data, "dataset_properties.pkl"), preprocessing_output_dir)
    # shutil.copy(join(raw_data, "dataset.json"), preprocessing_output_dir)
    # step.3 生成预处理计划
    # plan_resample(cropped_data)
    # step.4 重采样&标准化
    prepare_preprocess(args.tf) # 0.72656 0.72656 4


if __name__ == "__main__":
    main()