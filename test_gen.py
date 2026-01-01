# import torch
# import os

# # 高速化ライブラリの使用を強制的にオフにする設定
# os.environ["ACCELERATE_USE_XFORMERS"] = "FALSE"

# from diffusers import StableDiffusionPipeline

# model_id = "runwayml/stable-diffusion-v1-5"

# print("モデルのロードを開始します...")
# # CPUとGPUの橋渡しを一番シンプルな形で行う設定
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16,
#     safety_checker=None
# )
# pipe.to("cuda")

# prompt = "A red apple on the left of a blue cup"

# print("画像生成中...（この処理には1分ほどかかる場合があります）")
# # 高度な最適化を使わずに生成
# with torch.no_grad():
#     image = pipe(prompt).images[0]

# image.save("result_test.png")
# print("成功しました！ result_test.png を確認してください。")

# import torch
# import os
# os.environ["ACCELERATE_USE_XFORMERS"] = "FALSE"
# from diffusers import StableDiffusionPipeline

# model_id = "runwayml/stable-diffusion-v1-5"

# print("モデルをロード中...")
# # すでに一度ダウンロード済みなので、ここは一瞬で終わるはずです
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     safety_checker=None
# )
# pipe.to("cuda")

# # プロンプトをより具体的にして、AIの「空間認識力」を試します
# prompt = "a professional photo of a bright red apple on the left side of a blue porcelain cup, on a clean white table, studio lighting, 8k"

# print("画像生成中...")
# with torch.no_grad():
#     image = pipe(prompt).images[0]

# # 最終確認用の名前で保存
# image.save("result_final_step.png")
# print("\n🎉 生成成功！")
# print("左側のファイル一覧から 'result_final_step.png' を開いてみてください。")


# import torch
# import os
# os.environ["ACCELERATE_USE_XFORMERS"] = "FALSE"
# from diffusers import StableDiffusionPipeline

# model_id = "runwayml/stable-diffusion-v1-5"

# print("1. ベースモデルをロード中...")
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     safety_checker=None
# )
# pipe.to("cuda")

# print("2. CoMPaSS (SD1.5用軽量パッチ) を適用中...")
# # 著者のリポジトリからSD1.5用の重みを直接指定します
# # ※今回はエラーを避けるため、信頼性の高いHF公式形式で読み込みます
# try:
#     pipe.load_lora_weights("blurryg/CoMPaSS", weight_name="compass_sd15.safetensors")
#     print("CoMPaSSのロードに成功しました！")
# except Exception as e:
#     print(f"LoRAロード失敗: {e}")
#     print("リポジトリ名やトークン設定を確認してください。")

# # CoMPaSSの効果が出やすいプロンプト
# # 「左にリンゴ、右にコップ」という配置をAIに強く意識させます
# prompt = "a red apple on the left, a blue cup on the right, high quality"

# print("3. CoMPaSSを有効にして画像を生成中...")
# with torch.no_grad():
#     image = pipe(prompt).images[0]

# image.save("result_with_compass_sd15.png")
# print("\n🎉 生成成功！ 'result_with_compass_sd15.png' を確認してください。")


# import torch
# import os
# from diffusers import StableDiffusionPipeline

# # エラー回避のための設定
# os.environ["ACCELERATE_USE_XFORMERS"] = "FALSE"

# # あなたのHugging Faceトークンをここに貼り付けてください
# MY_TOKEN = ""

# model_id = "runwayml/stable-diffusion-v1-5"

# print("1. ベースモデル（SD1.5）をロード中...")
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     use_auth_token=MY_TOKEN,
#     safety_checker=None
# )
# pipe.to("cuda")

# print("2. CoMPaSS (SD1.5用) をダウンロード・適用中...")
# # 著者の最新のリポジトリ構成に合わせたパスを指定します
# # ※blurgyy/CoMPaSS-FLUX.1 という名前でもSD1.5用のファイルが含まれている場合があります
# try:
#     # 著者のリポジトリ名を確認し、LoRAファイルを読み込みます
#     # もしエラーが出た場合は、この repo_id を README にある正確なものに書き換えてください
#     pipe.load_lora_weights(
#         "blurgyy/CoMPaSS-FLUX.1", 
#         weight_name="compass_sd15.safetensors",
#         use_auth_token=MY_TOKEN
#     )
#     print("✅ CoMPaSSのロードに成功しました！")
# except Exception as e:
#     print(f"❌ LoRAロード失敗: {e}")
#     print("※このエラーが出ても、土台のAIだけで画像生成を試みます。")

# prompt = "a red apple on the left, a blue cup on the right, high quality"

# print("3. 画像を生成中...")
# with torch.no_grad():
#     image = pipe(prompt).images[0]

# image.save("result_compass_sd15_real.png")
# print("\n🎉 生成完了！ 'result_compass_sd15_real.png' を確認してください。")


# import torch
# import os
# from datetime import datetime
# from diffusers import StableDiffusionPipeline

# # --- 設定 ---
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
# # 試したいパッチの情報（SD1.5用）
# PATCH_REPO = "blurgyy/CoMPaSS-FLUX.1"
# PATCH_NAME = "compass_sd15"  # パッチ名として使用
# WEIGHT_FILE = "compass_sd15.safetensors"
# # MY_TOKEN = "あなたのトークンをここに貼る"
# MY_TOKEN = ""

# # PROMPT = "a red apple on the left, a blue cup on the right"
# PROMPT = "A pink cat on the left of a green dog, 8k"    # 左にピンクの猫、右に緑の犬

# # フォルダ作成（実行時刻で分ける）
# current_dir = os.path.dirname(os.path.abspath(__file__))
# output_root = os.path.join(current_dir, "outputs")
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = os.path.join(output_root, f"experiment_{timestamp}")

# os.makedirs(output_dir, exist_ok=True)
# print(f"📁 出力フォルダ: {output_dir}")

# # --- 1. ベースモデルの準備 ---
# print("1. ベースモデルをロード中...")
# pipe = StableDiffusionPipeline.from_pretrained(
#     MODEL_ID, 
#     torch_dtype=torch.float16,
#     safety_checker=None
# ).to("cuda")

# # --- 2. パッチなし (Before) の生成 ---
# print("2. [パッチなし] で生成中...")
# with torch.no_grad():
#     img_no = pipe(PROMPT).images[0]
#     img_no.save(f"{output_dir}/01_base_only.png")

# # --- 3. パッチあり (After) の生成 ---
# print(f"3. [パッチあり: {PATCH_NAME}] の合体を試行中...")
# try:
#     pipe.load_lora_weights(
#         PATCH_REPO, 
#         weight_name=WEIGHT_FILE,
#         use_auth_token=MY_TOKEN
#     )
#     print("✅ パッチの合体に成功！")
    
#     with torch.no_grad():
#         img_with = pipe(PROMPT).images[0]
#         # ファイル名にパッチ名を入れる
#         img_with.save(f"{output_dir}/02_with_{PATCH_NAME}.png")
#     print(f"✅ 保存完了: 02_with_{PATCH_NAME}.png")

# except Exception as e:
#     print(f"❌ パッチの適用に失敗しました: {e}")
#     print("※パッチファイルが取得できないため、比較画像は作成されませんでした。")

# print(f"\n実験終了。フォルダ '{output_dir}' を確認してください。")



# import torch
# import os
# from datetime import datetime
# from diffusers import StableDiffusionPipeline, UNet2DConditionModel
# from safetensors.torch import load_file

# # --- 設定 ---
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
# # アップロードしたファイルのパス（名前が違う場合はここを書き換えてください）
# COMPASS_WEIGHTS_PATH = "diffusion_pytorch_model.safetensors"

# PROMPT = "A pink cat on the left of a green dog, 8k"

# # フォルダ作成
# current_dir = os.path.dirname(os.path.abspath(__file__))
# output_root = os.path.join(current_dir, "outputs")
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = os.path.join(output_root, f"experiment_{timestamp}")
# os.makedirs(output_dir, exist_ok=True)

# print(f"📁 出力フォルダ: {output_dir}")

# # --- 1. ベースモデルのロード ---
# print("1. ベースモデルをロード中...")
# pipe = StableDiffusionPipeline.from_pretrained(
#     MODEL_ID, 
#     torch_dtype=torch.float16,
#     safety_checker=None
# ).to("cuda")

# # --- 2. パッチなし (Before) の生成 ---
# print("2. [CoMPaSSなし] で生成中...")
# with torch.no_grad():
#     img_no = pipe(PROMPT).images[0]
#     img_no.save(f"{output_dir}/01_base_only.png")
# print("✅ 保存完了: 01_base_only.png")

# # --- 3. CoMPaSS重みの注入 (After) ---
# print(f"3. [CoMPaSS重み] をUNetに注入中...")
# if os.path.exists(COMPASS_WEIGHTS_PATH):
#     try:
#         # safetensors形式の重みを読み込む
#         state_dict = load_file(COMPASS_WEIGHTS_PATH)
        
#         # モデルの心臓部(unet)の重みを、CoMPaSSのものに差し替える
#         pipe.unet.load_state_dict(state_dict)
#         print("✅ CoMPaSS重みの注入に成功しました！")
        
#         print("4. [CoMPaSSあり] で生成中...")
#         with torch.no_grad():
#             img_with = pipe(PROMPT).images[0]
#             img_with.save(f"{output_dir}/02_with_compass.png")
#         print("✅ 保存完了: 02_with_compass.png")
        
#     except Exception as e:
#         print(f"❌ 注入エラー: {e}")
# else:
#     print(f"❌ エラー: {COMPASS_WEIGHTS_PATH} が見つかりません。アップロードを確認してください。")

# print(f"\n実験終了。フォルダ '{output_dir}' を確認してください。")


import torch
import os
from datetime import datetime
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

# --- 設定 ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"
COMPASS_WEIGHTS_PATH = "diffusion_pytorch_model.safetensors"
PROMPT = "A pink cat on the left of a green dog, 8k"

# フォルダ作成
current_dir = os.path.dirname(os.path.abspath(__file__))
output_root = os.path.join(current_dir, "outputs")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(output_root, f"experiment_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

print(f"📁 出力フォルダ: {output_dir}")

# --- 1. ベースモデルのロード ---
print("1. ベースモデルをロード中...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# --- 2. パッチなし生成 ---
print("2. [CoMPaSSなし] で生成中...")
with torch.no_grad():
    img_no = pipe(PROMPT).images[0]
    img_no.save(f"{output_dir}/01_base_only.png")
print("✅ 保存完了: 01_base_only.png")

# --- 3. CoMPaSS重みの注入 ---
print("3. CoMPaSS重みを注入中...")
if os.path.exists(COMPASS_WEIGHTS_PATH):
    try:
        # 🌟 メモリを節約しながら読み込む設定
        state_dict = load_file(COMPASS_WEIGHTS_PATH, device="cuda") 
        
        # UNetの重みを差し替え
        pipe.unet.load_state_dict(state_dict)
        
        # 読み込み終わったら不要なメモリを即座に解放
        del state_dict
        torch.cuda.empty_cache()
        
        print("✅ CoMPaSSの注入に成功しました！")
        
        print("4. [CoMPaSSあり] で生成中...")
        with torch.no_grad():
            img_with = pipe(PROMPT).images[0]
            img_with.save(f"{output_dir}/02_with_compass.png")
        print("✅ 保存完了: 02_with_compass.png")
        
    except Exception as e:
        print(f"❌ 注入エラー: {e}")
else:
    print(f"❌ ファイル未完了 (現在サイズを確認してください)")

print(f"\n実験終了。")