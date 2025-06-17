import SimpleITK as sitk
import os
from os.path import join
import shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['WXSUPPRESS_SIZER_FLAGS_CHECK'] = '1'


def get_files_with_extension(path, extensions): #获取文件夹下后缀名.nii或.nii.gz的文件
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files

def check_nii_and_tar_unpack(folder_path):
    nii_gz_file = None
    tar_file = None
    # 遍历文件夹内的文件
    for file in os.listdir(folder_path):
        if any(file.endswith(ext) for ext in ['.nii.gz']): # ,'.nii'
            nii_gz_file = file
        elif file.endswith('.tar'):
            tar_file = file
    # 如果同时存在 .nii.gz 文件和 .tar 文件
    if nii_gz_file and tar_file:
        # 删除 .nii.gz 文件
        nii_gz_file_path = os.path.join(folder_path, nii_gz_file)
        if os.path.exists(nii_gz_file_path):
            os.remove(nii_gz_file_path)
            print(f"Deleted {nii_gz_file}")
        # 解压 .tar 文件到当前文件夹
        tar_file_path = os.path.join(folder_path, tar_file)
        shutil.unpack_archive(tar_file_path, folder_path)
        os.remove(tar_file_path)
        print(f"Extracted and Deleted {tar_file}")
    elif tar_file:
        # 解压 .tar 文件到当前文件夹
        tar_file_path = os.path.join(folder_path, tar_file)
        shutil.unpack_archive(tar_file_path, folder_path)
        os.remove(tar_file_path)
        print(f"Extracted and Deleted {tar_file}")
    else:
        print("No tar file, still nii or nii.gz file.")

def dicom2nii(dicom_seriesdir_path, save_nii_path):
    # 对于已经存在.nii格式的数据
    ct_file = get_files_with_extension(dicom_seriesdir_path, ['.nii'])
    if ct_file:
        print("get CT .NII file" + ct_file[0])
        ct = sitk.ReadImage(ct_file[0])
        sitk.WriteImage(ct, save_nii_path)
        return
    # 将文件夹下一系列dicom数据转换成nii格式
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_seriesdir_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, save_nii_path)

def resample_sitkImage(sitkImage, outspacing):
    #读取文件的size和spacing信息
    original_size = sitkImage.GetSize()
    original_spacing = sitkImage.GetSpacing()
    #计算改变spacing后的size，用物理尺寸/体素的大小
    new_size = [int(round(original_size[0] * (original_spacing[0] / outspacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / outspacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / outspacing[2])))]
    #设定重采样的一些参数, 线性插值
    resampled_img = sitk.Resample(sitkImage, new_size, sitk.Transform(),
                                  sitk.sitkLinear, sitkImage.GetOrigin(),
                                  outspacing, sitkImage.GetDirection(), 0.0,
                                  sitkImage.GetPixelID())
    return resampled_img


def main():
    # 本任务的数据处理是直接先将数据保存为未预处理的nii.gz文件 保存至E盘des_path文件夹内
    # id: 病人
    # tp: 病人该CT文件的时间点 基线0 1 2 3...
    # fup：病人tp时间点下的三个阶段 a-1 v-2 d-3
    src_path = r'F:\Experiment_Data\Task_Pancreas_Series\pancreas_new_data-2024-12-30\data_resec' # r'E:\HSX_projects\pancreatic_cancer\pancreatic ct time series 112' # "E:/HSX_projects/pancreatic_cancer/pancreatic ct time series 658"
    des_path = r'F:\Experiment_Data\Task_Pancreas_Series\pancreas_new_data-2024-12-30\resec_niigz' # r'E:\HSX_projects\pancreatic_cancer\pancreatic_ct_time_series_112_niigz' # "E:/HSX_projects/pancreatic_cancer/pancreatic_ct_time_series_658_niigz"
    people = ""
    tp = 0
    fup = ""
    for filename in os.listdir(src_path): # 遍历src_path文件夹下所有子文件 filename = id+time+CT 有用的信息取id
        id = filename.split("+",3)[0]  # "10365331"
        if people != id: # 那这就是一个新人 并且第一次是基线
            people = id
            tp = 0
        else: # 那这个人已经出现过了 这是一次随访
            tp += 1
        # 进入当前文件夹的子文件夹 filename[-1] = avd 或 123
        for ctname in os.listdir(src_path+"\\"+filename): #一个文件夹中有多个时期avd
            if ctname[-1] == '1' or ctname[-1] == 'A':
                fup = 'a'
            if ctname[-1] == '2' or ctname[-1] == 'V':
                fup = 'v'
            if ctname[-1] == '3' or ctname[-1] == 'D':
                fup = 'd'
            # 当前CT文件dicom转nii.gz
            new_file_name = id+"_"+str(tp)+"_"+fup+".nii.gz"
            cur_ct_file_name = join(src_path, filename, ctname)
            print(f"do case {new_file_name}...")
            dicom2nii(cur_ct_file_name, join(des_path, 'images', new_file_name))
            # 当前文件夹下mask文件 如果是只有.nii文件就是他，如果还有tar文件就需要解压再找nii.gz 重命名且保存为.nii.gz
            check_nii_and_tar_unpack(cur_ct_file_name)
            mask_file = get_files_with_extension(cur_ct_file_name, ['.nii.gz']) # '.nii',
            mask = sitk.ReadImage(mask_file[0])
            sitk.WriteImage(mask, join(des_path, 'masks', new_file_name))


if __name__ == "__main__":
    main() # 理论上只跑一次这个函数了 say bye bye