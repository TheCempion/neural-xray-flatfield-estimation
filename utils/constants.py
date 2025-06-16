SEED = 69
EPS = 1e-7
NUM_PCA_COMPONENTS = 28

PCA_FILE_NAME = "pca.pkl"

# set std for flatfields retrieved from mg-wire and spider hair (see preprocess.ipynb)
FF_STD_MIN_MG_WIRE = 0.011934471
FF_STD_MAX_MG_WIRE = 0.0454644
FF_STD_MIN_SPIDER_HAIR = 0.0063703246
FF_STD_MAX_SPIDER_HAIR = 0.013694693
FF_STD_MIN = 0.01
FF_STD_MAX = 0.025

# matplotlib
FIG_SIZE = 6  # implemented options: "default" | 6 | 7

HIST_NUM_BINS = 200
HIST_ALPHA = 1
HIST_SHAREX = False
HIST_SHAREY = False
CMAP = "gray"
CBAR_LABEL = "Intensity / A.U."

# notations
ANNOT_MODEL_IN = "Input"
ANNOT_MODEL_OUT = "Output"
ANNOT_MODEL_TARGET = "Target"
ANNOT_FFC_IN = f"{ANNOT_MODEL_IN} / {ANNOT_MODEL_OUT}"
ANNOT_FFC_TARGET = f"{ANNOT_MODEL_TARGET} / {ANNOT_MODEL_OUT}"

ANNOT_H_RAW = r"$H_{raw}$"
ANNOT_H_REAL = r"$H_{real}$"
ANNOT_H_CORR = r"$H_{corr}$"
ANNOT_H_CORR_DL = r"$H_{corr}^{DL}$"
ANNOT_H_CORR_PCA = r"$H_{corr}^{PCA}$"
ANNOT_F_CORR = r"$F_{corr}$"
ANNOT_F_CORR_DL = r"$F_{corr}^{DL}$"
ANNOT_F_CORR_PCA = r"$F_{corr}^{PCA}$"
ANNOT_MODEL_IN_MATH = r"$x$"
ANNOT_MODEL_OUT_MATH = r"$\hat{y}$"
ANNOT_MODEL_TARGET_MATH = r"$y$"
ANNOT_FF_REAL = r"$F_{real}$"
ANNOT_SFF = r"$F_{synth}$"
ANNOT_SFF_DL = r"$F_{synth}^{DL}$"
ANNOT_SFF_PCA = r"$F_{synth}^{PCA}$"


# thresholding
THR_IMSHOW_MAX = 1.5

FILE_EXT = "pdf"


def set_file_ext(ext: str) -> None:
    global FILE_EXT
    FILE_EXT = ext


MPL_USE_TEX = True

# training settings
DEFAULT_TRAIN_INPUT_SIZE = 512

GAN_REAL_LABEL = 1.0
GAN_FAKE_LABEL = 0.0

NUM_VALID_STEPS_IN_EPOCH = 1
MAX_NUM_STEPS_BETWEEN_VALID = 50

BCE_LOSS_REDUCTION_METHOD = (
    "mean"  # "none" would also make sense, but pix2pix actually used "mean"
)


# Dictionary key convention
PREF_MODEL_STATE_DICT = "state_dict_model"
PREF_OPTIM_STATE_DICT = "state_dict_optimizer"
PREF_LR_SCHEDULER_STATE_DICT = "state_dict_lr_scheduler"
