# Spinal_T2FS_Simulator

1. The model architecture is located in the model folder, where our STS model is in the MMGSN/syn_model_2D_T2FS.py file.

2. The utils folder includes code for data loading, 5-fold validation data splitting, dataset preprocessing, and model metric calculation.

3. train2D_mmgsn_AllFold0507.py is the training code using the entire training set, while train2D_mmgsn_5Fold0506_Fold1.py is for 5-fold cross-validation on the training set. You can specify which fold to validate on (fold_num = 1/2/3/4/5) and train on the remaining four folds.

4. Model weights can be downloaded from: https://pan.quark.cn/s/18430bb8ad89. You can implement and apply them to similar medical image generation tasks. evaluate_models.ipynb is the model inference code. After downloading the model weights, you can modify the corresponding paths and run it. Please ensure your data format is consistent with our example (./Data/your_regis_imgdata).


5.  If you encounter any issues such as the model code or model weights not matching the model architecture, please contact us at wangchenxi@pku.edu.cn.