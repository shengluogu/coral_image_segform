import os
import numpy as np
from PIL import Image

def colorize_masks_and_save(input_path, top_n=900):
    output_folder=f'{input_path}_visual'
    """
    支持输入文件夹或单张图片。
    将仅含0,1,2,3的PNG图片映射为彩色图并保存。
    """
    # 1. 创建输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 已创建彩色图片输出目录: {output_folder}")

    # 2. 识别输入类型
    target_files = []
    source_dir = ""

    if os.path.isfile(input_path):
        # 如果输入是单张图片
        if input_path.lower().endswith('.png'):
            source_dir = os.path.dirname(input_path)
            target_files = [os.path.basename(input_path)]
            print(f"📝 检测到单图输入: {target_files[0]}")
        else:
            print(f"❌ 错误：文件 '{input_path}' 不是 PNG 格式。")
            return
    elif os.path.isdir(input_path):
        # 如果输入是文件夹
        source_dir = input_path
        all_files = os.listdir(input_path)
        png_files = [f for f in all_files if f.lower().endswith('.png')]
        png_files.sort()
        target_files = png_files[:top_n]
        print(f"📝 找到 {len(png_files)} 张PNG，将处理前 {len(target_files)} 张...")
    else:
        print(f"❌ 错误：找不到路径 '{input_path}'")
        return

    # 3. 定义颜色映射表
    color_map = np.array([
        [0,   0,   0],    # 0: 黑色 (背景)
        [255, 50,  50],   # 1: 鲜红色 (活珊瑚)
        [50,  255, 50],   # 2: 鲜绿色 (死珊瑚)
        [50,  50,  255],  # 3: 鲜蓝色 (白化珊瑚)
    ], dtype=np.uint8)

    # 4. 遍历处理
    success_count = 0
    for filename in target_files:
        # 如果 source_dir 为空（当前目录文件），join 也能正常工作
        image_path = os.path.join(source_dir, filename)
        
        try:
            img = Image.open(image_path)
            mask_array = np.array(img)

            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]

            max_val = mask_array.max()
            if max_val > 3:
                print(f"⚠️ 警告：{filename} 最大值 {max_val} > 3，跳过。")
                continue

            # 核心上色
            color_img_array = color_map[mask_array]
            colorize_img = Image.fromarray(color_img_array)
            
            save_name = f"{filename}"
            save_path = os.path.join(output_folder, save_name)
            colorize_img.save(save_path)
            success_count += 1
            
        except Exception as e:
            print(f"⚠️ 处理 {filename} 时出错: {e}")

    print(f"\n🎉 处理完成！共生成 {success_count} 张彩色图片至 '{output_folder}'。")

if __name__ == "__main__":
    # 使用示例 1：处理整个文件夹
    colorize_masks_and_save('val_results') 

    # 使用示例 2：只处理单张图片
    #colorize_masks_and_save('results/mask_001.png')