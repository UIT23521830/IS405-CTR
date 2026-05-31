

Guan et al. *Journal of Big Data* (2025) 12:223  
https://doi.org/10.1186/s40537-025-01281-9

Journal of Big Data

RESEARCH

Open Access

# Accurate and interpretable CTR prediction via distilled neural additive feature interaction network

Fei Guan<sup>1\*</sup>, Jiahuan Zhan<sup>1</sup> and Jing Yang<sup>1</sup>

\*Correspondence:

Fei Guan

guanfei@hubei.edu.cn

<sup>1</sup>College of Statistics &

Mathematics, Hebei University

of Economics and Business,

Shijiazhuang, China

**Abstract**

Click-through rate (CTR) prediction, which estimates the probability that a user will click on an advertisement or an item, is critical to online advertising recommender systems. A key factor in optimizing CTR is understanding how to discover and explain uncommon or hidden feature interactions concealed behind user behaviors. In this paper, a neural additive feature interaction network model is firstly constructed (abbreviated as NAFI), which can automatically learn the low- and high- order feature interactions of input features with good explainability. Then a multi-teacher knowledge distillation network is utilized to realize the lightweight of NAFI (abbreviated as KD-NAFI). Finally, comprehensive experiments on three public datasets demonstrate the accuracy and interpretability of our models.

**Keywords** Click-through rate prediction, Neural additive model, Recommender systems, Deep neural network, Knowledge distillation

**Introduction**

The progressive shift from traditional to mobile digital advertising has elevated online advertising to a prominent position within the advertising industry. Predicting the likelihood of a user clicking on an advertisement or an item is known as click-through rate (CTR) prediction [1, 2], which is an important issue for an online advertising recommender system. CTR prediction is fundamentally a binary classification task. Over recent years, classification algorithms have found transformative applications beyond the digital domain, particularly driving significant advancements in medical image analysis, disease risk stratification, and so on [3, 4]. The expected advertising revenue is calculated as CTR×bid [5], where bid represents the revenue the system receives per user click on an advertisement. Consequently, since CTR is a core variable in the revenue model, even marginal improvements in its prediction accuracy can significantly enhance recommendation efficiency and increase business revenue.

In CTR prediction scenarios, features are typically sparse and multi-field categorical variables, necessitating sophisticated modeling of feature interactions for accurate

© The Author(s) 2025. **Open Access** This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit <http://creativecommons.org/licenses/by-nc-nd/4.0/>.

Guan et al. Journal of Big Data (2025) 12:223

Page 2 of 22

predictions. Consider an e-commerce clothing platform where item features include: Style = {casual, formal, sporty}, Color = {red, blue, green, ...}, Brand = {Nike, Adidas, Gucci, ...}, and Season = {spring, summer, fall, winter}. For instance, customers preferring casual Adidas clothing highlight the importance of modeling the interaction {Style = casual, Brand = Adidas} for providing relevant recommendations. Furthermore, color popularity often exhibits seasonal dependencies, making the interaction between the Color and Season fields particularly significant. These examples underscore the critical role of field-aware feature interaction modeling. Therefore, effectively modeling feature interactions and obtaining meaningful features is a popular research subject in CTR prediction.

While Factorization Machines (FMs) [6] and their variants effectively model pairwise interactions, they suffer from inherently high computational complexity for modeling all feature pairs. This scaling becomes a severe bottleneck in industrial-scale applications with massive feature spaces, significantly compromising both training efficiency and real-time inference latency. FM-based deep models address the interaction order limitation by incorporating Deep Neural Networks (DNNs) [7] to capture implicit high-order interactions. However, this introduces its own set of efficiency challenges: (a) The intrinsic complexity of DNNs themselves incurs substantial computational and memory overhead; (b) More critically, these models typically learn interactions implicitly and entangle them within dense hidden layers. This lack of explicit, structured modeling hinders the efficient identification and utilization of the most predictive, yet often sparse and field-specific, interaction patterns. Consequently, this can lead to suboptimal performance and redundant computation on irrelevant feature combinations.

Consequently, there remains a pressing need for a CTR prediction model that can: (1) Efficiently capture high-order feature interactions without high computational costs, and (2) Explicitly and effectively identify and model the most salient, sparse, field-aware feature combinations critical for accuracy in practice. To bridge this gap, we propose KD-NAFI, a novel model designed for high accuracy, superior efficiency, and enhanced interpretability. Our key contributions are:

- We create an accurate and interpretable neural additive feature interaction model (NAFI) with five components.
- We develop a KD-NAFI model via a multi-teacher knowledge distillation network.
- The validity of our proposed models has been verified by comparing the performance with many state-of-the-art models on three public datasets.

The rest of the paper is organized as follows. Sect. “**Related Work**” is a literature review of related work. Sect. “**Preliminaries**” concludes the preparatory knowledge of the models presented. Sect. “**KD-NAFI: model construction and algorithm implementation**” describes the basic architecture of our model NAFI and algorithm implementation of KD-NAFI. Sect. “**Experiments**” records some experimental results. We summarize and point out the possible future research directions in Sect. “**Conclusion and future work**”.

### **Related work**

While feature interaction modeling remains fundamental to CTR prediction, research methodologies span diverse approaches: Logistic Regression, Factorization Machine, Deep Learning, etc.

Guan et al. Journal of Big Data (2025) 12:223

Page 3 of 22

### **Logistic regression**

Early CTR prediction models primarily employed Logistic Regression (LR) [8]. As a classic data mining technique, LR derives predictive equations through historical data fitting. However, while widely adopted for its simplicity and effectiveness, LR suffers from two critical limitations: (1) Its assumption of feature independence inherently neglects feature interactions; (2) The model's optimal fit to training data causes significant accuracy degradation when applied to new data distributions. Consequently, developing robust feature interaction mechanisms is essential.

### **Factorization Machine-based models**

Rendle et al. proposed Factorization Machines (FMs) to capture second-order feature interactions—an approach proven effective for diverse tasks in online advertising systems. Based on FM, many variants were proposed [9–11]. Juan [12] developed the Field-aware Factorization Machines (FFMs) model, which allowed each feature to learn multiple vectors where each vector was associated with a domain. However, FFFMs were restricted by the need of large memory and cannot easily be used in internet companies. Xiao [13] presented the Attention Factorization Machine model (AFM) based on the attention mechanism to focus on the significance of distinct feature weights. However, the factorization machine-based models suffer from high time complexity. Consequently, despite their ability to model feature interactions, these methods are often computationally prohibitive in practice.

### **Deep Learning-based models**

As the rapid development of Deep Learning (DL), some Deep Learning-based models [14, 15] were proposed via Deep Neural Networks to learn complex high-order feature interactions through embedding vectors and nonlinear activation functions. Among them, Zhang [16] presented the Factorization machine supported Neural Networks (FNN) model, which leveraged FM pretraining results as the input of DNN. Nevertheless, this model overemphasized the interactions of higher-order features while neglecting the interactions of lower-order features. Cheng et al. [17] presented the Wide & Deep model, which combined a logical model and a depth model for better feature interactions and effectively improved CTR. Wang et al. [18] introduced the Deep & Cross Network (DCN), which extends FM-style interaction modeling through explicit feature crossing to capture higher-order dependencies. However, its unconstrained crossing mechanism generates extensive redundant interactions, compromising model efficiency. Guo et al. [19] presented the DeepFM model, which combines the FM and DNN models to accomplish higher and lower order feature interactions. Lian [20] proposed xDeepFM, leveraging compressed interaction networks for explicit higher-order feature modeling and deep neural networks for implicit feature learning. However, xDeepFM faces high computational complexity and parameter inefficiency. Song et al. [21] presented the AutoInt model based on a multi-head self-attention mechanism, which improved the interpretability while enhancing the prediction performance. Nevertheless, AutoInt's reliance on multi-head self-attention may lead to high computational overhead, potential overfitting due to excessive parameters, and limited ability to filter noisy or irrelevant feature combinations. Li [22] proposed the ACN model, which efficiently captured the diverse interests of users through their historical behaviors. Xie et al. [23] proposed the DRIN model,

Guan et al. Journal of Big Data (2025) 12:223

Page 4 of 22

which learned both implicit and explicit interactions of features and simulated explicit interactions between two layers of arbitrary order. Zhang et al. [24] thought of integrating user interest with attention mechanism and proposed the IARM, which combines deep neural networks to capture user interest while interacting with features to determine the final expression of interest. He et al. [25] proposed a new graph-based feature interaction model that combines the flexible and explicit representation capabilities of graphs with the learnability of FMs to improve the performance of the model. Lou et al. [26] proposed a highly usable CTR model which can simultaneously learn multiple granularities of user interests and implicit feature interactions. Wei et al. [27] presented a click conversion rate prediction model based on different user behaviors in a heterogeneous network. Yang et al. [28] proposed two models for the deep component. One is deep neural networks enhanced by multi-head self attention mechanism, the other computes high-order feature interaction by stacking multiple attention blocks. Recently, Vaghari et al. have proposed many recommendation models focusing on interpretability. For example, DiaRec [29] captures the temporal evolution of user intentions through a session-aware attention mechanism. GCORec [30] adopts a group-aware attention mechanism to hierarchically group features based on domain knowledge. HAN [31] leverages a hierarchical attention network to capture multi-level feature interactions.

The Deep Learning-based models mentioned above demonstrate that a significant issue in modeling high-order feature interactions within raw data is the large computational cost incurred as the number of original features rises. Thus, Zhu et al. [32] developed an integrated CTR prediction model by the knowledge distillation networks, which improved data utilization by adaptively assigning attentional weights and using a collection of teacher models for distillation to obtain a more accurate student model. Jose et al. [33] employed a gated network architecture. They configured the network to integrate and distill high-performing CTR models, aiming to derive a CTR model tailored for real-time recommender systems. Guan et al. [34] created a DICN model, which can realize higher and lower order feature interactions under varying weights. Additionally, they presented a distillation network-based approach that allowed the model to meet the requirements of online deployment. Sang [35] proposed a plug-and-play fusion self-distillation module composed of two parts. The first component established connections between explicit and implicit feature interactions, while the second component enhances the accuracy and robustness of the framework through self-hinting mechanisms.

Based on the above literatures, core challenges in CTR prediction involve: (1) Enabling transparent model reasoning—crucial for diagnosing bias, ensuring regulatory compliance, and refining feature engineering by transforming opaque decisions into actionable insights. However, current approaches face limitations: interpretable additive models (e.g., NAM [36]) fail to capture feature interactions; interpretable interactive models (e.g., NAFM [37]) are restricted to explicit low-order interactions; while black-box deep models (e.g., DNN) inherently lack transparent decision logic. (2) Balancing accuracy, interpretability, and complexity. Many DNN-based architectures demand intensive matrix computations, resulting in high model complexity. Different from the existing work, our model NAFI and KD-NAFI continue the pursuit of interpretability through structured attention, explicitly quantifying static feature interactions via additive structures and multi-head attention. They do not rely on temporal correlations or predefined groups, forming a complement to the models proposed by Vaghari et al. in terms of

Guan et al. Journal of Big Data (2025) 12:223

Page 5 of 22

scenario adaptability. This paper necessitates optimizing prediction accuracy under interpretability constraints while reducing complexity to enable lightweight deployment—thereby improving online service feasibility.

### Preliminaries

This Section establishes the theoretical groundwork for our proposed NAFI and KD-NAFI models. We first define the click-through rate (CTR) prediction task and provide its formal description to establish a common understanding of the problem. We then introduce the core concepts and methodologies underpinning our modeling approach: neural additive models, the transformer attention mechanism, and knowledge distillation. These methods respectively inform the design of the models' key components. This foundation directly supports the subsequent model architecture design.

### CTR prediction

The objective of click-through rate (CTR) prediction is to estimate a user clicking on an ad or an item in a specific scenario. It uses contextual features as input vectors, identified as  $x = (x_1, x_2, \dots, x_m)$ , where the number of features represented by  $m$ . The label  $y \in \{0, 1\}$  makes it clear whether or not to click on the advertisement or item, we then proceed to perform a search operation on every feature  $x_j$  and map it to  $d$ -dimensional embedding  $e_j \in \mathbb{R}^d$ . With this operation, the primal feature vector can be expressed as an embedded list of features  $(e_1, e_2, \dots, e_m)$ .

### Neural additive model

In 2020, Agarwal et al. presented the Neural Additive Model (NAM) (see Fig. 1), which trains several deep learning-based neural networks in an additive way. NAM extends the Generalized Additive Model [38] (GAM). Deep neural networks are employed by NAM to discern non-linear correlations among features and their interactions.

Where the formula for NAM is defined as follows:

$$g(E[y]) = \beta + f_1(x_1) + f_2(x_2) + \dots + f_k(x_k) \quad (1)$$

In (1),  $x_1, x_2, \dots, x_k$  are the inputs of the feature,  $y$  is the target variable,  $g(\cdot)$  is the feature function and represents the output. Each  $f_i$  is a univariate shape function and the individual input variables are processed by the corresponding neural network.

### Transformer model

Multi-head self-attention technique was presented in the Transformer model [39] (see Fig. 2).

It has been successfully applied in fields such as image captioning, text generation, and 3D model retrieval [40–42]. The last few years, the multi-head self-attention mechanism has achieved extraordinary success in modeling complicated interactions for its good interpretability. The multi-head self-attention [43] projects the input embedding matrix onto multiple subspaces, calculates the weight relationships for each subspace, and then aggregates them together to obtain the final attention coefficients.

$$y = \sigma(w^T(e_1^{\text{Res}} \oplus e_2^{\text{Res}} \oplus \dots \oplus e_m^{\text{Res}}) + b) \quad (2)$$

Guan et al. Journal of Big Data

(2025) 12:223

Page 6 of 22

**Fig. 1** NAM's structural diagram

In (2), *y* represents the gating scalar output,  $\sigma$  denotes the sigmoid activation function.  $c_i^{\text{Res}}$  refers to residual output of the *i*-th attention head,  $\oplus$  indicates vector concatenation.

**Knowledge distillation**

In 2015, Hinton [44] introduced the notion of Knowledge Distillation (KD), which matched large complex models by training a compact model with fewer parameters. Knowledge Distillation (see Fig. 3) is a method for lightweight neural networks [45], its basic concept is to use prior knowledge from a big model neural network (teacher network) to train a smaller but still superior performing compact neural network (student network). Knowledge Distillation improves the precision of the student model in mirroring the teacher's predictions [46].

$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)} \quad (3)$$

In (3), *q<sub>i</sub>* is the *i*th probability value output after distillation, and *T* is the distillation temperature, when *T*=1, the results are normal prediction results, and the prediction results will tend to be equal when *T* gradually becomes larger. Thus, during training, we first raise the temperature to reduce the variance in model predictions. Once the learning capability of the student model improves, we then set the temperature to 1 to ensure the accuracy of model predictions [47, 48].

Guan et al. Journal of Big Data

(2025) 12:223

Page 7 of 22

Padding  
Masking  
〈BOS〉  
〈EOS〉

Outputs  
Probabilities

Linear & SoftMax

Add & Norm

Point-wise FFN

Add & Norm

Multi-Head Attention

Add & Norm

Masked Multi-Head Attention

Positional Encoding

Encoder Embedding

Decoder Embedding

Inputs

Outputs  
(shifted right)

Fig. 2 Structural diagram of the transformer model

Teacher model  
input x  
Layer 1 Layer 2 ... Layer m  
Softmax(T=t) soft labels  
distillation loss Loss Fn  
Teacher model  
Layer 1 Layer 2 ... Layer m  
Softmax(T=t) soft predictions  
Softmax(T=1) hard prediction  
student loss Loss Fn  
hard label y

Fig. 3 Knowledge distillation structural diagram

### KD-NAFI: model construction and algorithm implementation

This Section details the design and implementation of KD-NAFI. We first describe the base NAFI architecture, then we present the knowledge distillation process used to derive KD-NAFI.

Guan et al. Journal of Big Data

(2025) 12:223

Page 8 of 22

### Basic architecture of NAFI model

In this Section, a novel Neural Additive Feature Interaction (NAFI) model is proposed based on the Neural Additive model (NAM) by incorporating a higher-order feature interaction module. In Fig. 4, the basic structure of NAFI comprises five components: a feature input layer, a feature embedding layer, a neural additive network (NAM) layer, a feature interaction layer (FIN), and an output layer. The feature input and embedding layers are similar to those in some classical CTR models. They use a sparse representation for input features and transform the raw feature input into a dense vector. The NAM layer learns the contribution of each individual feature to the output. The FIN layer captures higher-order feature interactions using a Multi-head Self-Attention mechanism. Finally, the outputs from NAM and FIN are combined to produce the final prediction score. A brief description of these key components is as follows.

#### Feature input and embedding layer

The feature input layer receives raw data and converts it into a feature representation that can be processed by the neural network. The feature embedding layer is capable of mapping discrete features into layers of a continuous low-dimensional space.

$$E_x = [V_1^T, V_2^T, \dots, V_i^T, \dots, V_h^T] \quad (4)$$

In Eq. (4),  $E_x$  denotes the vector representation of the feature,  $h$  indicates the total number of feature fields, and  $V_i \in R^k$  represents the  $i$ th embedded feature vector.

#### Neural additive network layer

This layer concentrates on the learning of a feature function for each feature through the utilization of a feature network [49], to represent the contribution of each feature to the output.

**Fig. 4** The basic architecture of NAFI model

Guan et al. Journal of Big Data

(2025) 12:223

Page 9 of 22

For a given observation, NAM represents the output of each feature network through a feature function, and the output contribution of each feature  $x_i$  can be obtained from the respective network. Thus the influence of each feature on the prediction is independent of the other features, leading to an interpretable model.

In NAM, exp-centered(ExU) was proposed instead of Relu as an activation function. The conclusion in [36] showed that NNs with ExU units are able to fit the toy dataset significantly better than standard NNs. Thus, we adopt ExU units following [36] to train accurate NAMs. They are more generally applicable for approximating jumpy functions with neural nets. For the input feature  $x$ , each hidden unit using an activation function  $f$  computes  $h(x)$  given by

$$h(x) = f(e^w * (x - b)) \quad (5)$$

Where  $w$  and  $b$  represent weight and bias parameters respectively. A hidden unit within the ExU function ought to be capable of significantly altering its output in response to a minuscule change in input.

The neural additive network layer independently models and explicitly outputs the nonlinear contribution of each feature through an additive structure. In CTR prediction, ad click decisions often rely on a few core static features. Unlike traditional models where feature contributions are often integrated into overall interactions—making it difficult to identify key factors individually—the NAM layer achieves feature contribution separation, enabling clear quantification of each feature's independent impact. This not only preserves the interpretability of low-order features but also provides a weighted input basis for subsequent interaction layers.

#### *Feature interaction layer*

After embedding features into a low-dimensional dense space, understanding individual feature contributions is essential. Additionally, modeling higher-order feature combinations within this space is crucial to identify and form meaningful higher-level features [50]. In the feature interaction layer, we address this issue using the Multi-head self attention mechanism [51] in the Transformer model. Furthermore, a feature may also involve several varying combinations of features, so in this paper, we determine which combinations of features are meaningful by using key-value attention [52], and also incorporate residual network [53] on this architecture to retain the previously learned effective features. The design of multi-head self-attention combined with residual networks differs from traditional methods that solely use multi-head attention for interaction modeling. Residual networks ensure that the low-order information of original features is not overshadowed by high-order transformations, thus preserving feature interpretability. Meanwhile, multi-head attention dynamically captures key high-order interactions, addressing the issue of weak generalization ability in modeling high-order interactions. Figure 5 shows the architecture of the feature interaction layer.

In Fig. 5,  $W_{query}^{(h)}$ ,  $W_{key}^{(h)}$  and  $W_{value}^{(h)}$  are the transformation matrices of the input vectors, the attention score of each feature is represented by  $\alpha_m^{(h)}$ . To perform the process both simply and efficiently, the dot product is used for computation, and then each set of self-attention is spliced together to perform a linear transformation to obtain the final output.

Thus the output formula is defined as follows:

Guan et al. Journal of Big Data

(2025) 12:223

Page 10 of 22

**Fig. 5** A multi-head attention structure map fusing residual ideas

$$\tilde{e}_m = \tilde{e}_m^{(1)} \oplus \tilde{e}_m^{(2)} \oplus \dots \oplus \tilde{e}_m^{(h)} \quad (6)$$

To preserve the learned combined features, we put standard residual connections to the network. The output is represented as:

$$e_m^{Res} = relu(\tilde{e}_m + W_{Res} e_m) \quad (7)$$

After processing through a feature interaction layer, each feature is transformed into a new representation capturing higher-order interactions. Multiple such layers can be stacked, where the output of one layer serves as input to the next, enabling modeling of arbitrary-order feature combinations.

**Output layer**

The output of NAFI is to fuse the output results of NAM and FIN, where NAM and FIN are parallel structures. The output result of NAM and FIN can be expressed using Eq. (8) and Eq. (9).

$$NAM(x) = w_{nam} + f_1(v_1) + f_2(v_2) + \dots + f_k(v_k) \quad (8)$$

$$FIN(x) = w_{fin} + \sigma(w^T (e_1^{Res} \oplus e_2^{Res} \oplus \dots \oplus e_m^{Res}) + b) \quad (9)$$

Where  $w_{nam}, w_{fin}$  are denoted as the deviation terms of NAM and FIN, respectively. The NAFI output is formed by combining the outputs of the NAM function and the FIN interaction.

$$NAFI(x) = sigmoid(NAM(x) + FIN(x)) \quad (10)$$

**KD-NAFI algorithm implementation**

As depicted in Fig. 1, NAM processes each input feature separately through individual neural networks and then combines their outputs. Consider NAM model with  $d$  input features, each processed by an  $L$ -layer neural network containing  $H$  neurons per layer. Consequently, Each feature network has  $H + (L - 2) \times H^2 + H$  parameters, resulting

Guan et al. Journal of Big Data (2025) 12:223

Page 11 of 22

in  $d \times (H + (L - 2) \times H^2 + H)$  total parameters for  $d$  features. So the amount of computation required for each feature is  $O(H^2 \times L)$ . Since it is an independent neural network, each feature can be computed in parallel, the overall complexity of NAM is  $O(d \times H^2 \times L)$ . Thus, NAFI will exhibit a high level of complexity. Following we will employ a knowledge distillation framework to develop its lightweight counterpart KD-NAFI. This strategic approach effectively mitigates computational complexity while maintaining model efficacy, thereby achieving a more efficient architecture design.

The algorithm flow of KD-NAFI is presented below:

**Step 1** Train and adapt the existing independent CTR models, set the logloss function as their objective function.

**Step 2** Integrate the trained CTR model, evaluate the accuracy of each integrated model.

**Step 3** Select the integrated model with the highest AUC values as the teacher model.

**Step 4** Obtain the student model by distilling the parameters obtained from training in a large integrated teacher model.

We propose an adaptive model ensembling framework. It dynamically weights models to resolve structural heterogeneity and feature processing divergence. Therefore, ensemble weights are dynamically computed using model predictions. The weighting function is defined as:

$$\alpha_i = \frac{\exp(f(Z_i))}{\sum_i^k \exp(f(Z_i))} \quad (11)$$

where  $\alpha_i$  is the weight of each model,  $Z_i$  is the  $i$ th teacher model, and  $f$  is the mapping function learned by the teacher.

### Experiments

This Section describes the experiments in detail, containing the experimental setup, model performance comparison, interpretability experiments and the ablation study.

#### Experiment settings

In this paper, all experimental studies were conducted on a unified computing platform featuring the following technical specifications: The software stack comprised Python 3.7 with PyTorch and TensorFlow deep learning frameworks, supported by CUDA 11.1 acceleration libraries on Windows 10 OS. The hardware configuration consisted of an NVIDIA GTX 1050 GPU with 16GB system RAM, ensuring consistent computational resource allocation throughout the experiments.

The experimental parameters were uniformly configured across all models to ensure fair comparison. Following the settings in references [54, 55], we adopted the Adam optimizer with cross-entropy loss as the objective function. The learning rate was dynamically selected from {0.0001, 0.001, 0.01} through automated tuning. For embedding layers, we fixed the vector dimension  $k$  at 16 while maintaining other hyperparameters consistent with their original implementations. In DNN-enhanced models, the hidden architecture consisted of three fully-connected layers with 32 neurons per layer. Dataset-specific batch sizes were configured as follows: 4096 for Criteo, 2048 for Avazu, and 128 for MovieLens-1 M. Hyperparameter spaces were defined with dropout rates ranging from 0.1 to 0.9 and L2 regularization coefficients selected from {1e-5, 1e-4, 1e-3}.

Guan et al. Journal of Big Data (2025) 12:223

Page 12 of 22

We evaluate our proposed models on the following three benchmark datasets. Table 1 summarizes the datasets.

**Criteo** This dataset is the most commonly used in click-through rate prediction. It has 13 integer features and 26 categorical features, with a high cardinality in each category.

**Avazu** This dataset includes user mobile behavior over a 10-day period, encompassing user clicks on display mobile ads. It consists of 23 feature fields that encompass user attributes, device data, and targeted ad features, with a total of 40 million observations.

**MovieLens-1M:** This dataset comprises over 1 million ratings from over 6000 individuals for over 4000 movies. There are three tables: user information, movie information, and a rating table. When the three tables are combined, we can get the following user information: user age, gender, occupation, movie title, subject to which the movie belongs, and movie rating.

Performance evaluation for CTR prediction is conducted exclusively on the Criteo and Avazu datasets, the MovieLens-1 M dataset is excluded from performance metrics calculation. For interpretability experiments involving visualization analysis, the Avazu and MovieLens-1 M datasets are utilized. Both datasets possess well-defined metadata, enabling clear visualization of feature weights and interaction relationships.

Criteo and Avazu datasets are divided in an 8:1:1 manner, i.e., 80% for the training set and 10% for both the validation and test sets. **It is worth noting that an increase in AUC or a decrease in Logloss even at the level of 0.001 is considered meaningful for click-through rate prediction, which has been stated in the existing surveys [16–18, 56].**

**Evaluation metrics**

In this paper, AUC and Logloss are selected as the criteria for evaluating the validity of our models.

$$AUC = \frac{\sum_{ins \in \text{positive class}} rank_{ins} - \frac{M \times (M+1)}{2}}{M \times N} \quad (12)$$

In Eq. (12),  $M$  denotes the number of positive samples and  $N$  denotes the number of negative samples. A higher AUC value indicates better performance of prediction.

In order to learn the weights and parameters of the model, in this paper, the Logarithmic loss function (Logloss) is used as the objective function, and it is expressed as follows:

$$L = -\frac{1}{N} \sum_{i=1}^{N} (y_i \log(\sigma(\hat{y}_i)) + (1 - y_i) \log(1 - \sigma(\hat{y}_i))) \quad (13)$$

**Table 1** Datasets information

| Data          | recording | feature-field | Sparsity |
|---------------|-----------|---------------|----------|
| Criteo        | 45 M      | 39(26+13)     | 96.62%   |
| Avazu         | 40 M      | 23            | 95.74%   |
| MovieLens-1 M | 739,012   | 7             | 95.50%   |

Guan et al. Journal of Big Data (2025) 12:223

Page 13 of 22

The Logloss is a loss function used to evaluate the predictive accuracy of a classification model, the lower the Logloss value, the better the model performance.

In addition, we also use the RelaImpr metric to represent the model's enhancement, which is calculated by the following formula:

$$RelaImpr = \left( \frac{AUC_{measured\_mod\_el} - 0.5}{AUC_{base\_mod\_el} - 0.5} - 1 \right) \times 100\% \quad (14)$$

RelaImpr reflects a more intuitive picture of the model's optimization ratio. Higher RelaImpr values indicate a better model.

### **Baseline models**

**LR** [8]. LR models a binary dependent variable using the basic form of a logistic function. **FM** [6]. FM models the second-order feature interaction using factorization technology to achieve this goal.

**AFM** [13]. AFM incorporates an attention mechanism on top of FM, which gains the importance of features, but only to the second order.

**DNN** [7]. DNN consists of embedding and MLP layers for CTR prediction.

**DCN** [18]. DCN builds a special cross-network to display modeling interaction features.

**DeepFM** [19]. DeepFM is an end-to-end learning model that integrates FM and DNN architectures. It uses FM to model low-order feature interactions and DNN to model higher-order feature interactions.

**xDeepFM** [20]. xDeepFM includes the CIN network, capable of learning higher-order feature interactions and modeling features implicitly and explicitly.

**AutoInt** [21]. AutoInt learns feature interactions and captures higher-order interactions between features automatically.

**NAM** [36]. NAM exploits deep learning-based neural networks to capture nonlinear relationships between features.

**DICN** [34]. DICN can realize higher and lower order feature interactions under varying weights.

**IARM** [24]. IARM combines deep neural networks to capture user interest while interacting with features to determine the final expression of interest.

**Attention Enhanced DeepFM** [28]. There are two models. One is deep neural networks enhanced by multi-head self attention mechanism, the other computes high-order feature interaction by stacking multiple attention blocks.

**NAFM** [37]. NAFM combines the features of the neural additive model (NAM) and FM to enhance the performance of the model.

### **Prediction experiments and performance comparison**

We compare the performance of all the above models in Table 2. FM is used as the baseline model to react to the improvement. Meanwhile, we conduct a two-tailed T-test to assess the statistical significance between our models and the baseline. (★:  $P < 1e-3$ ).

The above results show that: (1) The performance of NAM is comparable to that of LR model for the fact that neither model takes feature interactions into account. This similarity demonstrates that relying solely on first-order features is insufficient for accurate CTR prediction. However, NAM outperforms LR, which validates the effectiveness of neural additive networks in enhancing model performance despite the lack of explicit

Guan et al. Journal of Big Data

(2025) 12:223

Page 14 of 22

**Table 2** Comparison of different models on Criteo and Avazu datasets

| Model                      | Criteo  |                |                | Avazu        |                |                |              |
|----------------------------|---------|----------------|----------------|--------------|----------------|----------------|--------------|
|                            | AUC     | Logloss        | Relalmpr       | AUC          | Logloss        | Relalmpr       |              |
| LR                         | 0.7851  | 0.4663         | -2.73%         | 0.7536       | 0.3945         | -5.72%         |              |
| FM(base)                   | 0.7931  | 0.4582         | 0.00%          | 0.7690       | 0.3754         | 0.00%          |              |
| AFM                        | 0.7959  | 0.4550         | 0.96%          | 0.7761       | 0.3819         | 2.64%          |              |
| DNN                        | 0.8002  | 0.4526         | 2.42%          | 0.7752       | 0.3827         | 2.30%          |              |
| DCN                        | 0.8048  | 0.4483         | 3.99%          | 0.7777       | 0.3811         | 3.23%          |              |
| DeepFM                     | 0.8028  | 0.4513         | 3.31%          | 0.7763       | 0.3815         | 2.71%          |              |
| xDeepFM                    | 0.8068  | 0.4480         | 4.67%          | 0.7782       | 0.3808         | 3.42%          |              |
| AutoInt                    | 0.8071  | 0.4457         | 4.78%          | 0.7775       | 0.3831         | 3.16%          |              |
| NAM                        | 0.7901  | 0.4588         | -1.02%         | 0.7600       | 0.3850         | -3.35%         |              |
| DICN                       | 0.8073  | 0.4428         | 4.84%          | 0.7792       | 0.3801         | 3.79%          |              |
| IARM                       | 0.7941  | 0.4844         | 0.34%          | 0.7720       | 0.3931         | 1.12%          |              |
| Attention Enhanced Deep FM | Model 1 | 0.8031         | 0.4503         | 3.41%        | 0.7769         | 0.3814         | 2.94%        |
|                            | Model 2 | 0.8042         | 0.4462         | 3.79%        | 0.7774         | 0.3842         | 3.12%        |
| NAFM                       |         | 0.8069         | 0.4482         | 4.71%        | 0.7771         | 0.3809         | 3.01%        |
| NAFI(our model)            |         | <b>0.8092*</b> | <b>0.4426*</b> | <b>5.49*</b> | <b>0.7801*</b> | <b>0.3751*</b> | <b>4.13%</b> |

consideration for feature interactions. (2) NAFI outperforms all the other baseline models. The results on the two datasets are 5.49% and 4.13% higher than FM, demonstrating that NAFI has a large improvement in prediction performance. (3) Compared to NAM, the performance of the Relalmpr metrics of NAFI on the two datasets is improved by 6.58% and 7.73%, which shows that incorporating high-order feature interactions into a neural additive network is effective. Thus it is necessary to capture effective high-order feature interactions. (4) Model performance variations stem from differences in architectures and dataset characteristics. On the Criteo dataset, xDeepFM exhibits a lower AUC than AutoInt. This is likely because Criteo's feature interactions are dominated by low-order or shallow high-order patterns. xDeepFM's Compressed Interaction Network (CIN) focuses on explicit high-order modeling, which risks generating excessive invalid combinations that dilute predictive signals. In contrast, AutoInt's parameter-efficient design concentrates computation on adaptive attention weights and embeddings, making it more lightweight and better suited to Criteo's moderate sparsity and interaction complexity. Conversely, on the Avazu dataset, xDeepFM outperforms AutoInt in AUC. This reversal can be attributed to Avazu's reliance on explicit high-order feature combinations for ad exposure prediction. xDeepFM's CIN excels at capturing such patterns through vector-level hierarchical crosses, while AutoInt's self-attention mechanism struggles with sparsity-induced weight dilution-high-dimensional categorical features disperse attention scores, weakening focus on critical interactions.

**Interpretability experiments and result analysis**

To verify the validity of NAFI in interpretability, we used the Avazu and the Movielens-1 M datasets to plot the feature weights(See Fig. 7, 8, 9 and 10, because all the features in Criteo dataset are anonymized). In the figures, the darker the color of the corresponding cell position of the feature, the higher the score is, indicating that the feature plays a greater role in the final decision-making process.

In the Avazu dataset, it is noticeable that some features have sample records, and all of them are categorical features except for anonymous features such as C1 and C14-C21, where the features from 0-21 are as follows: 'hour', 'C1', 'banner\_pos', 'site\_id',

Guan et al. Journal of Big Data

(2025) 12:223

Page 15 of 22

'site\_domain', 'site\_category', 'app\_id', 'app\_domain', 'app\_category', 'device\_id', 'device\_ip', 'device\_model', 'device\_type', 'device\_conn\_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'.

We can see from Fig. 6 that the weights of the individual features of the Avazu dataset, and the 'device\_model' and 'device\_conn\_type' exhibits the highest weights. In addition, the domain and category of the website or application along with certain anonymous features, also hold significance.

Figure 7 presents the weight interaction diagram for each feature within the Avazu dataset. Notably, a positive correlation exists between banner\_pos and site\_domain. Additionally, site\_category and site\_domain demonstrate a strong correlation, which aligns with common knowledge. Features such as device\_type, C1, site\_id, site\_category, and certain anonymous features display substantial interactions among themselves. Conversely, device\_ip and device\_id show a relatively weak correlation with other attributes. This stability arises because they primarily function as user identity proxies, making them less susceptible to influence from other attributes.

In the Movielens-1 M data set, as shown in Fig. 8, the combined features output by NAFI, the part marked by the red box, indicates that males aged 18–24 prefer films with action and adventure themes, and this result is reasonable.

We also examined the correlation among various feature domains. That is, the correlation between feature domains is measured by the attention score of feature domains in the data. The result of the correlation between different feature domains is shown in Fig. 9.

Figure 9 shows that the parts marked by red boxes are strongly correlated, i.e., gender and movie attributes, age and movie attributes, occupation and movie number, and gender, age and movie attributes. It is reasonable to show a strong correlation between these

**Fig. 6** Schematic diagram of feature weights

Guan et al. Journal of Big Data

(2025) 12:223

Page 16 of 22

hour  
C1  
banner\_pos  
site\_id  
site\_domain  
site\_category  
app\_id  
app\_domain  
app\_category  
device\_id  
device\_ip  
device\_model  
device\_type  
device\_conn\_type  
C14  
C15  
C16  
C17  
C18  
C19  
C20  
C21  
hour  
C1  
banner\_pos  
site\_id  
site\_domain  
site\_category  
app\_id  
app\_domain  
app\_category  
device\_id  
device\_ip  
device\_model  
device\_type  
device\_conn\_type  
C14  
C15  
C16  
C17  
C18  
C19  
C20  
C21

**Fig. 7** Schematic diagram of feature interaction weights

feature domains, because the user's age and gender influence their preferences for movies, and their professional attributes also affect the themes they favor, it indicates that these characteristics are key determinants in whether a user enjoys a particular film. It effectively verifies that NAFI has good interpretability.

### Distillation experiment and result analysis

It is evident that NAFI exhibits strong predictive capabilities and interpretability. However, due to the neural additive network within the model, which results in NAFI having high complexity. Therefore, following we will employ a multi-teacher knowledge distillation network to achieve the lightweight design of NAFI as depicted in Sect. "KD-NAFI Algorithm Implementation".

Firstly, the top five performing models are ensembled using weights to verify the integration results of different teacher models. The four models—DCN, DeepFM, xDeepFM and AutoInt—have demonstrated superior performance. Ensemble Models 1–4 are each constructed by integrating a unique combination of three out of the four base models. Ensemble Model 5 is then derived by integrating all four base models (See Table 3), then observe the results of the AUC values of each integrated model.

The prediction results of the five integrated models and the distilled models are shown in Fig. 10. We can see that ensemble Model 3 has the largest AUC value on Avazu and Criteo datasets. Therefore, during the distillation process, Model 3 is selected as the teacher model, representing a weighted integration of DeepFM, xDeepFM, and AutoInt (denoted as Ensemble-CTR) while the student model is NAFI. The distilled model is denoted as KD-NAFI. All results are shown in Table 4.

Guan et al. Journal of Big Data

(2025) 12:223

Page 17 of 22

**Fig. 8** Single-feature interaction diagram

The results in Table 4 reveal that the integration model outperforms the single model in terms of efficiency, attaining the highest AUC. This finding indicates that combining sub-models with excellent performance can effectively boost prediction capabilities. Specifically, the KD-NAFI model, after implementing a knowledge distillation network, demonstrates higher prediction accuracy than the NAFI model alone. This outcome shows that during the distillation process, the student model has effectively absorbed certain capabilities of the teacher model, thus enhancing its own prediction accuracy.

To further validate the outstanding performance of KD-NAFI, Fig. 11 demonstrates the runtime of the different models on the Criteo and Avazu datasets.

As seen in Fig. 11, LR is the more time-efficient algorithm due to its simplicity. FM and NAM perform similarly in terms of runtime. The other models increase their running time due to the incorporation of deep learning modules, and it can be seen that the more complex the model is, the longer the running time it takes.

A summary of the results presented in Table 4; Fig. 11 indicates that KD-NAFI is a robust lightweight model, providing commendable accuracy at a low computational cost.

**The ablation study**

To investigate the roles of different components in KD-NAFI model, in this Section, we conduct the ablation study. We compare the KD-NAFI model with several variants: (1) One without the NAM component(FIN). (2) One without the FIN component(NAM).

Guan et al. Journal of Big Data

(2025) 12:223

Page 18 of 22

**Fig. 9** Feature domain interaction diagram

**Table 3** Model integration comparison

| Ensemble Models  | DNN | DCN | DeepFM | xDeepFM | AutoInt |
|------------------|-----|-----|--------|---------|---------|
| Ensemble Model 1 |     | ✓   |        | ✓       | ✓       |
| Ensemble Model 2 |     | ✓   | ✓      | ✓       | ✓       |
| Ensemble Model 3 |     |     | ✓      | ✓       | ✓       |
| Ensemble Model 4 |     | ✓   | ✓      | ✓       | ✓       |
| Ensemble Model 5 | ✓   | ✓   | ✓      | ✓       | ✓       |

**Fig. 10** AUC values of integrated prediction and distilled integrated prediction

Guan et al. Journal of Big Data

(2025) 12:223

Page 19 of 22

**Table 4** Comparison of distillation model predictions on Criteo and Avazu datasets

| Model        | Criteo        |               |          | Avazu         |               |          |
|--------------|---------------|---------------|----------|---------------|---------------|----------|
|              | AUC           | Logloss       | Relalmpr | AUC           | Logloss       | Relalmpr |
| DeepFM       | 0.8028        | 0.4513        | 3.31%    | 0.7763        | 0.3815        | 2.71%    |
| xDeepFM      | 0.8068        | 0.4480        | 4.67%    | 0.7782        | 0.3808        | 3.42%    |
| AutoInt      | 0.8071        | 0.4457        | 4.78%    | 0.7775        | 0.3831        | 3.16%    |
| Ensemble-CTR | 0.8122        | 0.4399        | 6.52%    | 0.7836        | 0.3426        | 5.43%    |
| NAFI         | 0.8092        | 0.4426        | 5.49%    | 0.7801        | 0.3797        | 4.13%    |
| KD-NAFI      | <b>0.8105</b> | <b>0.4409</b> | 5.94%    | <b>0.7806</b> | <b>0.3752</b> | 4.31%    |

**Run time per epoch**

**criteo**

**Run time per epoch**

**avazu**

**Fig. 11** Plot of run times for each model

**Table 5** Prediction results of ablation study

| Model   | Implication                      | AUC(Criteo) | Relalmpr | AUC(Avazu) | Relalmpr |
|---------|----------------------------------|-------------|----------|------------|----------|
| NAM     | without the FIN component        | 0.7901      | 0.00%    | 0.7600     | 0.00%    |
| FIN     | without the NAM component        | 0.8071      | 5.86%    | 0.7775     | 6.73%    |
| NAFI    | without the distillation network | 0.8092      | 6.58%    | 0.7801     | 7.73%    |
| KD-NAFI | Add distillation network         | 0.8105      | 7.03%    | 0.7806     | 7.92%    |

(3) One without the KD(NAFI). We conducted the ablation experiments on the Criteo and Avazu datasets.

The ablation study results presented in Table 5 suggest that both the NAM and FIN components are essential for enhancing the performance of the NAFI model in CTR prediction tasks. Furthermore, the KD architecture enhances the accuracy, training efficiency, and convergence speed of the NAFI model. Therefore, the integration of all three

Guan et al. *Journal of Big Data*

(2025) 12:223

Page 20 of 22

components results in superior performance compared to employing any single component in isolation.

### **Conclusion and future work**

In this paper, we initially present a Neural Additive Feature Interaction (NAFI) model, capable of thoroughly learning both effective low- and high-order feature interactions. Then, the lightweight nature of NAFI is achieved through a multi-teacher knowledge distillation network. Experimental results on three real-world data sets demonstrate the efficiency and good explainability of our proposed models. To capture more effective information for online advertising recommender systems, in the subsequent work, we will consider integrating self-supervised learning techniques for feature interaction modeling and delve deeply into the significance of attention mechanisms.

### **Acknowledgements**

This work is funded by Natural Science Foundation of Hebei Province of China (No. A2023207002), '333 Talent Project' of Hebei Province (No. C20221021), Key Program of Hebei University of Economics and Business (2024ZD12, 2023ZD10), Youth Team Support Program of Hebei University of Economics and Business.

### **Author contributions**

Fei Guan: Supervision, Resources, Project administration, Conceptualization. Jiahuan Zhan: Writing-original draft, Visualization, Software, Methodology. Jing Yang: Writing-review & editing, Validation, Formal analysis. All authors reviewed the manuscript.

### **Funding**

Not applicable.

### **Data availability**

No datasets were generated or analysed during the current study.

### **Declarations**

#### **Ethics approval and consent to participate**

Not applicable.

#### **Consent for publication**

Not applicable.

#### **Competing interests**

The authors declare no competing interests.

Received: 17 January 2024 / Accepted: 7 September 2025

Published online: 26 September 2025

### **References**

1. Wang X. A survey of online advertising click-through rate prediction models, Proceedings of 2020 IEEE International Conference on Information Technology, Big Data and Artificial Intelligence, 2020;516–521.
2. Tian Z, Bai T et al. EulerNet: adaptive feature interaction learning via Euler's formula for CTR prediction, proceedings of the 46th International ACM SIGIR conference on research and development in information retrieval, 2023;23–27.
3. Pololu N, Rajaram A. Transformation with Yolo tiny network architecture for multimodal fusion in lung disease classification. Cybernetics Syst. 2024;1–22. <https://doi.org/10.1080/01969722.2024.2343992>.
4. Aruna Kumar A, Bhagat A, Kumar S, Henge. Classification of diabetic retinopathy severity using deep learning techniques on retinal images. Cybernetics Syst. 2024;1–25. <https://doi.org/10.1080/01969722.2024.2375148>.
5. Jiang D, Xu R, Xu X, Xie Y. Multi-view feature transfer for click-through rate prediction. Inf Sci. 2021;546:961–76.
6. Rendle S. Factorization machines, Proceedings of IEEE International Conference on Data Mining, 2010;995–1000.
7. Hinton GE, Salakhutdinov RR. Reducing the dimensionality of data with neural networks. Science. 2006;313(5786):504–7.
8. Francq P, Fouss F. Introduction Recommender Syst, 2011, pp. 1–35.
9. Chen C, Xia F, Tong Z et al. Gradient boosting factorization machines, Proceedings of the 8th ACM Conference on Recommender systems, 2014;265–272.
10. Hong F, Huang D, Ge C. Interaction-aware factorization machines for recommfender systems, Proceedings of the AAAI Conference on Artificial Intelligence, 2019;3804–3811.
11. Xu C, Wu M. Learning feature interactions with lorentzian factorization machine, Proceedings of the AAAI Conference on Artificial Intelligence, 2020;6470–6477.
12. Juan Y, Zhuang Y, Chin WS et al. Field-aware factorization machines for CTR prediction, Proceedings of the 10th ACM Conference on Recommender Systems, 2016;43–50.

Guan et al. Journal of Big Data

(2025) 12:223

Page 21 of 22

13. Xiao J, Ye H et al. Attentional factorization machines: learning the weight of feature interactions via attention networks, Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, 2017;3119–3125.
14. Chen WQ, Zhan LZ et al. FLEN: leveraging field for scalable CTR prediction, 2019, arXiv preprint arXiv:1911.04690.
15. Huang P, He XD et al. Learning deep structured semantic models for web search using click through data, Proceedings of the 22nd ACM international conference on Information & Knowledge Management, 2013;2333–2338.
16. Zhang WN, Du TM, Wang J. Deep learning over multi-field categorical data, Proceedings of Advances in Information Retrieval, 2016;45–57.
17. Cheng HT, Koc L, Jeremiah H et al. Wide & deep learning for recommender systems, Proceedings of the 1st Workshop on Deep Learning for Recommender Systems, 2016;7–10.
18. Wang RX, Fu B, Fu G, Wang ML. Deep & cross network for ad click predictions, Proceedings of the ADKDD'17. New York, USA: Association for Computing Machinery, 2017;1–7.
19. Guo HF, Tang RM et al. DeepFM: a factorization-machine based neural network for CTR prediction, Proceedings of International Joint Conference on Artificial Intelligence, 2017;1–7.
20. Lian JX, Zhou XH et al. xDeepFM: combining explicit and implicit feature interactions for recommender systems, Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery&Data Mining, 2018;1754–1763.
21. Song WP, Shi ZP et al. AutoInt: automatic feature interaction learning via self-attentive neural networks, Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019;1161–117.
22. Li DF, Hu BT, et al. Attentive capsule network for click-through rate and conversion rate prediction in online advertising. Knowl.-Based Syst. 2021;211:106–522.
23. Xie J, Zhao XD, et al. Deep recurrent interaction network for click-through rate prediction. Inf Sci. 2022;604:210–25.
24. Zhang W, Han Y, Yi B, et al. Click-through rate prediction model integrating user interest and multi-head attention mechanism. J Big Data. 2023;10(1):1–15.
25. He QL, Zhou F, Gu LY, et al. A novel graph-based feature interaction model for click-through rate prediction. Inf Sci. 2023;649:119615.
26. Lou JG, Qin RZ, et al. Combining multi-interest activation and implicit feature interaction for CTR predictions. IEEE Trans Comput Social Syst. 2024;11(2):2889–900.
27. Wei SH, Zhang J, et al. A click conversion rate model of E-commerce platforms aiming at effective data sparse. IEEE Trans Emerg Top Comput Intell. 2024;8(2):1744–55.
28. Yang B, Liang J, et al. Study on interpretable click-through rate prediction based on attention mechanism. Comput Sci. 2023;50(5):12–20.
29. Vaghari H, Aghdam MH, Emami H. Diarec: dynamic intention-aware recommendation with attention-based context-aware item attributes modeling. J Artif Intell Soft Comput Res. 2024;14(2):171–89.
30. Vaghari H, Aghdam MH, Emami H. Group attention for collaborative filtering with sequential feedback and context aware attributes. Sci Rep. 2025;15(1):10050.
31. Vaghari H, Aghdam MH, Emami H. HAN: hierarchical attention network for learning latent context-aware user preferences with attribute awareness. IEEE Access, 2025.
32. Zhu JM, Liu JY, et al. Ensembled CTR prediction via knowledge distillation. ACM; 2020. pp. 2941–58.
33. Jose A, Shetty SD. Accurate and scalable CTR prediction model through model distillation. Expert Syst Appl. 2022;193:116474.
34. Guan F, Qian C, He FY. A knowledge distillation-based deep interaction compressed network for CTR prediction. Knowl.-Based Syst. 2023;275:110704.
35. Sang L, Ru Q et al. Feature interaction fusion self-distillation network for CTR prediction, arxiv Preprint arxiv:2411.07508v2, 2024.
36. Agarwal R, Frosst N, et al. Neural additive models: interpretable machine learning with neural Nets. ICML Workshop on Human Interpretability in Machine Learning; 2020.
37. Jose A, Shetty SD. Interpretable click-through rate prediction through distillation of the neural additive factorization model. Inf Sci. 2022;617:91–102.
38. Hastie TJ, Tibshirani RJ. Generalized additive models, Chapman and Hall/CRC, 1990.
39. Vaswani A, Shazeer N et al. Attention is all you need. Adv Neural Inf Process Syst, 2017, pp. 5998–6008.
40. Shao Z, Han JG et al. Textual context-aware dense captioning with diverse words. IEEE Trans Multimedia, <https://doi.org/10.1109/TMM.2023.3241517>
41. Shao Z, Han JG, et al. End-to-end dense captioning via multi-scale transformer decoding. IEEE Trans Multimedia. 2024;26:7581–93.
42. Chang JC, Zhang LY, Shao Z. View-target relation-guided unsupervised 2D image- based 3D model retrieval via transformer. Multimedia Syst. 2023;29(6):3891–901.
43. Lu WT, Yu YT, Chang YZ et al. A dual input-aware factorization machine for CTR prediction, in: 29th International Joint Conference on Artificial Intelligence and Seventeenth Pacific Rim International Conference on Artificial Intelligence, 2020;3139–3145.
44. Hinton G, Vinyals O, Dean T. Distilling the knowledge in a neural network. NIPS, 2015;1–9.
45. Choudhary T, Mishra V, Goswami A, Sarangapani J. A comprehensive survey on model compression and acceleration. Artif Intell Rev. 2020;53(3):5113–55.
46. Anil R, Pereira G, Passos A et al. Large scale distributed neural network training through online distillation, ICLR, Vancouver. arXiv preprint arXiv:1804.03235, 2018.
47. Kang SK, Hwang J, Kweon W, Yu H. DE-RRD: a knowledge distillation framework for recommender system, International Conference on Information and Knowledge Management, 2020;605–614.
48. Srinivas S, Fleuret F. Knowledge transfer with jacobian matching, In: 35th International Conference on Machine Learning, 2018;4723–4731.
49. Hattab MW, DeSouza RS, et al. A case study of hurdle and generalized additive models in astronomy: the escape of ionizing radiation. Monthly Notices of the Royal Astronomical Society; 2018. pp. 3307–21.
50. Beutel A, Covington P et al. Latent cross: making use of context in recurrent recommender systems, Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, 2018;46–54.

Guan et al. *Journal of Big Data*

(2025) 12:223

Page 22 of 22

- 51. Huang TW, Zhang ZQ, Zhang JL. FIBINET: combining feature importance and bilinear feature interaction for click-through rate prediction, Proceedings of the 13th ACM Conference on Recommender Systems, 2019;169–177.
- 52. Miller A, Fisch A et al. Key-value memory networks for directly reading documents, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, 2016;1400 – 140.
- 53. He KM, Zhang XY, Ren SQ, Sun J. Deep residual learning for image recognition, Proceedings of the IEEE conference on computer vision and pattern recognition, 2016;770–778.
- 54. Wang Z, She Q, Zhang J. MaskNet: introducing feature-wise multiplication to CTR ranking models by instance-guided mask. ArXiv Preprint arXiv:2102.07619, 2021.
- 55. Long LJ, Yin YF, Huang FL. Hierarchical attention factorization machine for CTR prediction, Proceedings of 27th International Conference on Database Systems for Advanced Applications, 2022;343–358.
- 56. Zhu JM, Liu JY, Yang S, Zhang Q, He XQ. Open benchmarking for click through rate prediction. Association for Computing Machinery; 2021. pp. 2759–69.

**Publisher's note**

Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.