# Explainability in Machine Learning Models for Credit Approval


## 1. Abstract

Defaulting on credit can significantly impact your ability to borrow money in the future. In this study, we found that a prior default de-
creases the chance of getting credit by 70 percent. This explanation is based on the analysis of three machine learning models (A, B, and
C) which are train on the Credit Approval Dataset. The study also found that each model emphasizes different features such as Credit
Score in Model A, Income in Model B, and Employed in Model C to decide about approving credit. This information is valuable
for consumers who are looking to get approved for new lines of credit. By understanding how defaults can impact your ability to
borrow money, you can take steps toward improving your score and increasing your chances of being approved.


## 2. Model A



### 2.1 Model Characteristics

In the first step, to inspect the model's decision boundary, which is a Support Vector Machine for a binary classification task, I visualized the decision regions of the model for pairs of features in 2 dimensions (Figure 1) . As a binary classification task, because all dataset features are fairly easily separable and normalized between zero and one, the SVM performs almost equally well and scores around 0.867 on the test dataset. However, <b> their respective decision boundary looks different from each other so that they cannot be separated with a single straight line meaning SVM is a type of Non-linear SVM </b>.


<p align = "center"><img src='./Model A/images/ModelA_DecisionBoundary.svg' width="700" height="250"></p>
<p align = "center"> Figure 1: SVM Decision Boundary (Model A) </p>

In the second step, to investigate the impact of features on the decision boundaries of the support vector machine, I used a Partial Dependence Plot per feature. PDP tells us how each feature contributes to the prediction. It also reveals which features are more important for accurate predictions. PDPs for Employed, Years Employed, Income, Debt, Credit Score, and Prior Default are shown in Figure 2. The most significant differences can be seen in the Prior Default. A credit default is a situation in which a borrower fails to make payments on their debt obligations. If <b>one has defaulted in the past, the probability that its credit will be confirmed drops significantly by about 70 percent</b>.



<table align = "center">
  <tr>
    <td><img img src='./Model A/images/ModelA_PDP_Credit_Score.svg' width="500" height="150"></td>
    <td><img src='./Model A/images/ModelA_PDP_Debt.svg' width="500" height="150"></td>
    <td><img src='./Model A/images/ModelA_PDP_Employed.svg' width="500" height="150"></td>
    <td><img img src='./Model A/images/ModelA_PDP_Income.svg' width="500" height="150"></td>
    <td><img src='./Model A/images/ModelA_PDP_Years_Employed.svg' width="500" height="150"></td>
    <td><img src='./Model A/images/ModelA_PDP_Prior_Default.svg' width="500" height="150"></td>
  </tr>
  </table>
   <p align = "center"> Figure 2: Partial Dependence Plot per feature (Model A) </p>



In the third step, <b>the partial dependence plot for two features that interact with respect to the outcome</b> is presented in two different ways. First, in Figure 3, Prior Default against Employed, Years Employed, Income, Debt, and  Credit Score are plotted in 3D. Second, the heatmaps in Figure 4. 

<table align = "center">
  <tr>
    <td><img img src='./Model A/images/ModelA_PDP_Prior_Default and Credit_Score Interaction DPD.svg' width="500" height="250"></td>
    <td><img src='./Model A/images/ModelA_PDP_Prior_Default and Debt Interaction DPD.svg' width="500" height="250"></td>
    <td><img src='./Model A/images/ModelA_PDP_Prior_Default and Employed Interaction DPD.svg' width="500" height="250"></td>
    <td><img img src='./Model A/images/ModelA_PDP_Prior_Default and Income Interaction DPD.svg' width="500" height="250"></td>
    <td><img src='./Model A/images/ModelA_PDP_Prior_Default and Years_Employed Interaction DPD.svg' width="500" height="250"></td>
  </tr>
  

 </table>
 <p align = "center">Figure 3: Two features interacting PDP (Model A) </p>

Figure 3 shows that there is <b>a linear relationship (Hyper-plane) between Prior Default and all other features, namely Employed, Years Employed, Income, Debt, Credit Score</b>. Any change from one to zero in the Prior Default significantly increases the predicted probability of getting approved for credit beyond the values of other features. Suppose the Prior Default has a value of one, the predicted probability decreases. Between the Prior Default feature and all other features, there is an interaction.

<table align = "center">
  <tr>
    <td><img img src='./Model A/images/ModelA_PDP_Prior_Default_Credit_Score.jpg' width="500" height="250"></td>
    <td><img src='./Model A/images/ModelA_PDP_Prior_Default_Debt.jpg' width="500" height="250"></td>
    <td><img src='./Model A/images/ModelA_PDP_Prior_Default_Employed.jpg' width="500" height="250"></td>
    <td><img img src='./Model A/images/ModelA_PDP_Prior_Default_Income.jpg' width="500" height="250"></td>
    <td><img src='./Model A/images/ModelA_PDP_Prior_Default_Years_Employed.jpg' width="500" height="250"></td>
  </tr>
  </table>
 <p align = "center">Figure 4: Two features interacting PDP (Model A) </p>
 
 
In the next step, Permutation Feature Importance was used to explain the importance of each feature in getting approved with the support vector machine (Table 1). As seen in Table 1, the most important feature was Prior Default, and the least important was Employed.



<table align = "center">
  <tr>
    <td>Name</td>
     <td>Importance</td>
     <td>Importance STD</td>
  </tr>
  <tr>
    <td>Prior Default </td>
    <td>0.3738  </td>
    <td> +/- 0.034</td>
  </tr>
  <tr>
    <td> Credit Score  </td>
    <td>0.0005 </td>
    <td>  +/- 0.001 </td>
  </tr>
  <tr>
    <td> Debt </td>
    <td> 0.0004  </td>
    <td> +/- 0.001</td>
  </tr>
    <tr>
    <td> Income </td>
    <td> 0.0003  </td>
    <td> +/- 0.001</td>
  </tr>
    <tr>
    <td> Years Employed </td>
    <td> 0.0002  </td>
    <td> +/- 0.001</td>
  </tr>
      <tr>
    <td> Employed </td>
    <td> 0.0001  </td>
    <td> +/- 0.001</td>
  </tr>
 </table>
  <p align = "center"> Table 1: Permutation Feature Importance (Model A)</p>


In the last step, I have utilized the library SHAP (Shapley Additive exPlanations) for my analyses and used SHAP Summary Plot (Figure 5) to detect  the greatest influence on the predictions for the two classes. The SHAP Summary Plot (Figure 5) indicates that Prior Default had the greatest influence on the predictions for all two classes, followed by Credit Score. This makes sense - these are two of the most important features in predicting whether someone will be approved for credit or not. 

<p align = "center"><img src='./Model A/images/SHOP_Summary_plot.png'  width="500" height="250"></p>
<p align = "center"> Figure 5: SHAP Summary Plot (Model A) </p>

To Summarize what I explored:


* <b>The SVM</b> is a type of Non-linear SVMs. 
* <b>Prior Default of 1</b> significantly increases the likelihood that the credit will not be approved by about 70 percent (first explanation).
* <b>Any change</b> from one to zero in the Prior Default increases significantly the predicted probability of getting approved for credit beyond the values of other features.
* <b>A low credit score</b> increases the likelihood of having a prior default.
* <b>Prior Default and Credit Score</b> are the two most important features to the final result (confirmed by Figure 2 and Table 1). 

### 2.2 Data Instance Analysis

LIME was first used to explain each individual prediction of three instances for this section. Figure 6 presents the explanations of each three instances based on LIME. 

<table align = "center">
  <tr>
    <td><img img src='./Model A/images/ModelA_X_test_inctances1.png' width="500" height="100"></td> 
    <td><img src='./Model A/images/ModelA_X_test_inctances2.png' width="500" height="100"></td>
    <td><img src='./Model A/images/ModelA_X_test_inctances3.png' width="500" height="100"></td>
  </tr>
  </table>
 <p align = "center">Figure 6: LIME on each three instances (Model A) </p>
 
 Figure 6 show that the most important feature for all three instances is Prior Default, which significantly affects the prediction. Based on my assumption in section 1,<b>I changed the prior default value from 1 to 0 or 0 to 1 in all data instances resulting in a reverse decision (Figure 7) </b>. 
 
 <table align = "center">
  <tr>
    <td><img img src='./Model A/images/ModelA_X_test_inctances1_1To0.png' width="500" height="100"></td> 
    <td><img src='./Model A/images/ModelA_X_test_inctances2_0To1.png' width="500" height="100"></td>
    <td><img src='./Model A/images/ModelA_X_test_inctances3_1To0.png' width="500" height="100"></td>
  </tr>
  </table>
 <p align = "center">Figure 7: Impact of change in the Prior default value of all data instances on LIME (Model A)</p>

 To put them succinctly:
 

* <b>The first instance</b> that has a prior default value of 1, the model does not approve credit with a probability of 0.78 percent. If only the prior default value changes from 1 to 0, the credit will be approved by the model with a probability of 0.93 percent. 

* <b>The second instance</b>, which has a prior default value of 0, the model will approve the credit with a probability of 0.93 percent. If only the prior default value changes from 0 to 1, the model will not approve the credit with a probability of 0.78 percent.   

* <b>The third instance</b>, which has a prior default value of 1, the credit will not also be approved by the model with a probability of 0.78 percent. If only the prior default value changes from 1 to 0, the credit is approved by the model with a probability of 0.93 percent.  


## 3. Model B


In the first step, to investigate the decision boundaries of model B, which is also a Support Vector Machine (SVM) for a binary classification task, Accumulated local effect was used to detect how features influence the prediction of the model on average. Figure 8 presents ALE plots for the approval prediction model by Employed, Years Employed, Income, Debt, Credit Score, and Prior Default. The Prior Default has a strong effect on the prediction. The average prediction significantly rises with changing Prior Default from 1 to 0. Income has a positive effect: When above 1 percent, the higher the relative Income, the higher the prediction. The Employed and Debt do not affect the predictions much.

<p align = "center"><img src='./Model B/images/ModelB_AccumulatedLocalEffects.png' width="600" height="250"></p>
<p align = "center"> Figure 8: Accumulated Local Effects for SVM (Model B) </p>

## 4. Model C

In the first step, to investigate the decision boundaries of the model, which is a deep multi-layer perceptron (MLP) classifier for a binary classification task, I plotted the decision ranges of the model for feature pairs in 2 dimensions. Since all features in the dataset are relatively easily separable and normalized between zero and one, the MLP classifier performs almost equally well, with a score of 0.867 for the test dataset.<b>As seen in Figure 9, the respective decision boundaries are similar so that they can be separated with a single straight line, which means that the MLP classifier belongs to linear classifiers</b>.


<p align = "center"><img src='./Model C/images/ModelC_DecisionBoundary.svg' width="600" height="250"></p>
<p align = "center"> Figure 9: Decision Boundary for MLP classifier (Model C)</p>
