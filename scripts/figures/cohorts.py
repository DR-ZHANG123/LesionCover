"""各图共用的队列与模型名称、颜色（只含显示设置）。"""
COHORTS = [("ddr_test", "DDR test"), ("deepdrid", "DeepDRiD"), ("idrid", "IDRiD"), ("messidor2", "Messidor-2"),
           ("eyepacs", "EyePACS"), ("mmrdr", "MMRDR")]
EXTERNAL = COHORTS[1:]
COH_LABEL = dict(COHORTS) | {"ddr_train": "DDR training", "ddr_dev": "DDR development"}
COH_COLOR = {"ddr_train": "#9E9E9E", "ddr_dev": "#BDBDBD", "ddr_test": "#424242", "deepdrid": "#1B9E77",
             "idrid": "#D95F02", "messidor2": "#7570B3", "eyepacs": "#E7298A", "mmrdr": "#66A61E"}
LC_COL = "#C0392B"
# 主比较中的模型：08s 输入名 → (显示名, 颜色, 标记)
MODELS = [("LesionCover", "LesionCover", LC_COL, "o"),
          ("DINOv2 global tokens only", "DINOv2-L, global view only", "#E8A39B", "o"),
          ("Global 1024", "ConvNeXt-T", "#4C72B0", "s"),
          ("Swin-T 1024", "Swin-T", "#55A868", "D"),
          ("EfficientNetV2-M 1024", "EfficientNetV2-M", "#8172B3", "^"),
          ("ResNet-50 1024", "ResNet-50", "#937860", "v"),
          ("GDRNet", "GDRNet", "#17BECF", "P"),
          ("CABNet", "CABNet", "#BCBD22", "X")]
MODEL_LABEL = {k: l for k, l, _, _ in MODELS}
MODEL_COLOR = {k: c for k, _, c, _ in MODELS}
MODEL_MARK = {k: m for k, _, _, m in MODELS}
LESIONS = [("MA", "Microaneurysm"), ("HE", "Haemorrhage"), ("EX", "Hard exudate"), ("SE", "Soft exudate")]
LES_COLOR = {"MA": "#D62728", "HE": "#8C2D04", "EX": "#E6AB02", "SE": "#1F78B4"}
