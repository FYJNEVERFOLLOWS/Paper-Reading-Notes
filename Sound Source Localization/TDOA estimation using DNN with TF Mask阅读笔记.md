# Introduction

[前人的工作是用DNN结合 TF masking（用mask把噪声干扰先过滤掉）通过某位置是否存在说话人的极大似然估计过滤来进行声源定位。而且都是仅仅训练mask，然后用得到的 IBM / IRM 和 GCC-PHAT 相乘再去计算 TDOA，作者认为应该训练 DNN 来估计 TDOA 的值，而不是只得到 TF mask。]

现有的方法都是每句话说完才预测出 TF mask，而作者的方法是实时性，可以逐帧估计TDOA 和 TF mask。作者对 TDOA 的估计是用回归而不是分类，所以会产生连续值而不是离散值。

作者提出的DNN结构的第一部分用于学习产生 TF mask，第二部分将预测得到的 TF mask 应用到 GCC-PHAT 来消除噪声的干扰并估计（和说话人位置相关的）TDOA 的值。该模型可以检验不同学习策略对 TF masking based TDOA 估计的影响。

# Signal Model and TDOA Estimation

$x_i(t,k)$: 第 i 个麦克风采集到的声音信号的时频表示，$k=0,...,K-1$是离散频率下标，$t$ 是帧下标

麦克风 i 和 i' 之间的 TDOA 表示为 $\tau_{ii'}^n=\tau_i^n-\tau_i'^n$（第 n 个声源位置$p_n$和第 i 个麦克风$m_i$之间的传播时长$\tau_i^n=||m_i-p_n||·c^{-1}$）

使用GCC-PHAT（广义互相关-相位变换方法）来计算时延

![image-20210825213932353](https://tva1.sinaimg.cn/large/008i3skNgy1gttd68nq35j60sa05ejrz02.jpg)

再将这个结果乘上mask

![image-20210825214018879](https://tva1.sinaimg.cn/large/008i3skNgy1gttd71c3kcj60nc01imx602.jpg)

在 t 时刻估计到的 TDOA 就是公式4的结果

![image-20210825214204671](https://tva1.sinaimg.cn/large/008i3skNgy1gttd8vh0ccj60kc02g0so02.jpg)

# Proposed Method

**Input Features**

该方法用(GCC-PHAT的值)空域信息和幅度谱作为 DNN 的输入特征来估计说话人的 TDOA。实际使用为了减少内存消耗使用频带而不是DFT。

空域特征通过以下步骤得到：

梅尔滤波器组乘上对第 k 个 frequency bin 的 GCC-PHAT 的结果

![image-20210825215232415](https://tva1.sinaimg.cn/large/008i3skNgy1gttdjr8xrlj60p601udfv02.jpg)
$$
R_{ii'}(\tau,t,b)=W(k,b)·R_{ii'}(\tau,t,k)
$$
这里的 · 表示矩阵相乘

再用 mask $\eta(t,b)$ 乘上上面这个结果然后在频带上积分得到 $R_{ii'}^m(\tau,t)$

把这个空域特征（后称为 masked GCC-PHAT），输入到 TDOA 估计的子网络中预测得到最终的 TDOA 值。

幅度谱特征是麦克风阵列梅尔频带对数幅度值的平均：$\log_{10}|x(t,b)|$



**DNN Achitecture for TDOA Estimation Using Masking**

![image-20210825222436248](https://tva1.sinaimg.cn/large/008i3skNgy1gttey990tpj60ry0h4wgs02.jpg)

用 LSTM cell 的原因是前一个 TDOA 的输出能用到下一个值的预测，避免杂散噪声和干扰产生的峰值带来的影响，这样预测得到的说话人 TDOA 更平滑。



**DNN Training Approaches**

The below approaches (A)-(D) to train the proposed DNN architecture were experimented with and compared to direct approach (E).

(A) **Implicit mask training:** Train using only the TDOA output. Output: TDOA.

(B) **Joint training:** Train mask prediction and the TDOA prediction simultaneously. Output: TF mask and TDOA.

(C)**Explicit mask training:** First train the TF mask prediction layers and then freeze their weights while training the TDOA estimation layers. 50% of the training data is used by mask prediction and TDOA prediction respectively. Output: TF mask and TDOA.

(D) **No masking:** Omit the masking process and train to predict the TDOA from GCC-PHAT values integrated over the frequency range. Output: TDOA.

(E) **Direct approach:** The masking stage is omitted, while the model inputs are kept the same. See Fig. 1b. Output: TDOA.

选择 MAE 作为 loss func ，同样是为了减轻 TDOA 异常值的影响而没有选择 MSE

![image-20210826115458131](https://tva1.sinaimg.cn/large/008i3skNgy1gtu1wax1tuj60eh0d0wgq02.jpg)

**TF Mask for Training**

mask = 信号分量与带噪音混响的观察值之比
$$
\eta_{i}(t, k)=\frac{\left|x_{\mathrm{i}, \mathrm{dp}}(t, k)\right|}{\left|x_{i}(t, k)\right|} \cos \left(\theta_{i}\right)
$$

# Simulated and Real-data

Simulated signals: TIMIT convolved with synthetic RIRs(Room Impulse Response) in a cuboid shaped room[7, 6.8, 3] m

训练集用间距分别是25、26、27、28、29、30cm的6麦采集到的立体声

验证集和测试集用间距分别是10、14、18、22、26的5麦采集

采集时用48kHz的采样率，最终处理时所有音频都降采样到16kHz

![image-20210826115530194](https://tva1.sinaimg.cn/large/008i3skNgy1gtu1wuj3i0j60es0bbwfr02.jpg)

![image-20210826115717662](https://tva1.sinaimg.cn/large/008i3skNgy1gtu1ypzv9cj60u60azdip02.jpg)

# Conclusions

 This paper proposed a DNN architecture that used magnitude spectrum information to derive a TF mask that was then applied to the spatial features (GCC-PHAT). The masked GCC-PHAT was then integrated over the frequency range and used to predict the TDoA value of the microphone pair. 



main contributions: 

A DNN architecture and training variant that learned to only predict the TDoA using implicit TF masking outperformed other variants that additionally learned to predict the oracle mask. Also outperformed a direct DNN for TDoA estimation without the TF masking stage and frequency integration, with 30 times less parameters.

现有的方法都是每句话说话才预测出 TF mask，而作者的方法是实时性，可以逐帧估计TDOA 和 TF mask。作者对 TDOA 的估计是用回归而不是分类，所以会产生连续值而不是离散值。

