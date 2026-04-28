import os
import torch
import tensorrt as trt
from torchreid import models

# ⭐ 新增：ONNX 相關（一定要）
import onnx
from onnx import external_data_helper

# ─── Config ─────────────────────────────────────
PT_WEIGHTS    = "weight/mobilenetv2_x1_0_market1501.pt"
ONNX_MODEL    = "weight/mobilenetv2_x1_0_market1501.onnx"
ENGINE_MODEL  = "weight/mobilenetv2_x1_0_market1501.engine"
INPUT_NAME    = "images"
OUTPUT_NAME   = "output"
INPUT_SHAPE   = (1, 3, 256, 128)   # 靜態輸入 shape，用於 ONNX export
OPSET_VER     = 12
WORKSPACE     = 1 << 30            # 1GB workspace
USE_FP16      = True              # True 可開 FP16（硬體需支援）
NUM_CLASSES   = 751                # Market1501 訓練 ID 數
# ────────────────────────────────────────────────

def build_reid_model():
    # 1. 建模型
    model = models.build_model(
        name="mobilenetv2_x1_0",
        num_classes=NUM_CLASSES,
        loss="softmax",
        pretrained=False
    )

    # 2. 手動 load checkpoint（去掉 module. 或 model. 前綴）
    ckpt = torch.load(PT_WEIGHTS, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("model."):
            name = name[len("model."):]
        new_state[name] = v
    model.load_state_dict(new_state)
    model.eval()
    return model

def export_onnx(model):
    dummy = torch.randn(INPUT_SHAPE)

    torch.onnx.export(
        model, dummy, ONNX_MODEL,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET_VER,
        do_constant_folding=True,
        # 只允許 batch 維動態，因此只針對第 0 維設定
        dynamic_axes={INPUT_NAME:{0:"batch_size"}, OUTPUT_NAME:{0:"batch_size"}}
    )
    print(f"[OK] ONNX 已儲存到：{ONNX_MODEL}")

    # ⭐⭐⭐ 關鍵：強制轉回 single-file ONNX
    model_onnx = onnx.load(ONNX_MODEL)
    external_data_helper.convert_model_from_external_data(model_onnx)
    onnx.save(model_onnx, ONNX_MODEL)

    print("[OK] ONNX external data 已合併為單一檔案")

def build_tensorrt_engine():
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, logger)

    # 1. parse ONNX
    with open(ONNX_MODEL, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX 解析失敗")

    # 2. 建 config 與 Optimization Profile
    config = builder.create_builder_config()
    # 設定 workspace 大小
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE)
    # 建立並加入 profile，這裡只支援 batch_size=1
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME,
                      min=(1,3,256,128),
                      opt=(1,3,256,128),
                      max=(1,3,256,128))
    config.add_optimization_profile(profile)
    # FP16
    if USE_FP16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] 已啟用 FP16")

    # 3. build serialized engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Serialized engine 建置失敗")

    # 4. 寫檔
    with open(ENGINE_MODEL, "wb") as f:
        f.write(serialized_engine)
    print(f"[OK] TensorRT engine 已儲存到：{ENGINE_MODEL}")

if __name__ == "__main__":
    if not os.path.isfile(PT_WEIGHTS):
        raise FileNotFoundError(f"權重檔 {PT_WEIGHTS} 不存在，請先下載並放到此資料夾")
    # 1. load model & weights
    model = build_reid_model()
    # 2. export ONNX
    export_onnx(model)
    # 3. build TensorRT engine
    build_tensorrt_engine()
