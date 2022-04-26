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


<p align = "center"><img src='./Model A/images/ModelA_DecisionBoundary.svg' alt="the satellite images and the corresponding labels" width="1000" height="500"></p>
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

