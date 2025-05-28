# Causal Story of Ecommerce Churn-out

### Team Members
- Anisha Thakrar
- Dyuthi Vinod
- Kavya Gajjar

### Objective & Motivation

For sustainable growth, the primary focus in E-Commerce business is increasing the Customer Acquisition and Customer Retention. 
To achieve that it should have low Churn out rate. Fundamentally, churn occurs when a customer stops consuming from a company. A high churn rate equals a low retention rate. Churn affects the size of your customer base and has a big impact on your customer lifetime value.

The motivation behind selecting this project was because E-Commerce websites have gained a lot of traction in the recent years. Every individual today has entered into the world of online shopping, including us. Therefore, we thought it would be interesting to analyse the causal effect on such website and to actually see how certain variables affect the churn rate of customers. The main goal of the project  is to find the factors that cause the customers to churn out

### Dataset

- The dataset used here is a publicly available [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) dataset
- It consists of 5630 rows and 20 columns
- The target variable here is **ChurnFlag** where if a customer has churned out, the value will be 1, else it is 0
- Treatment variables: SatisfactionFlag, ComplainFlag

*SatisfactionScore = 1 is SatisfactionFlag = 1; SatisfactionScore = [2,5]  is SatisfactionFlag = 0

### Assumptions

- Satisfaction Flag has two values True or False. If a customer is dissatisfied, they are more likely to churn out
- Complain is a sign of dissatisfaction, which might lead to churning out
- Higher value of days since last order indicates that customer hasn’t used the e-commerce website/app for a longer period of time (lower order count), hence more likely to churn out
- Lower order count in last month indicates less usage of the e-commerce website/app, hence lower % increase of orders from last year
- Increase in Percentage increase in orders from last year indicates the customer is active. Which means rate of churning out will be low
- Tenure is the length of relationship of the customer with the organization. Longer the tenure indicates more activity on the website, hence higher number of Order counts and higher Percentage increase in orders from last year
- City Tier; tier 1 might have higher number of order counts, which might lower the chance of churning out

### Video Link 

Video link to our presentation: https://youtu.be/C6TPtmsYvL4 


# 电商客户流失的因果分析

### 团队成员
- Anisha Thakrar
- Dyuthi Vinod  
- Kavya Gajjar

### 目标与动机

对于可持续发展，电商业务的主要关注点是提高客户获取和客户留存率。

为了实现这一目标，应该保持低流失率。从根本上说，当客户停止在某公司消费时就会发生流失。高流失率等于低留存率。流失会影响客户群规模，并对客户生命周期价值产生重大影响。

选择这个项目的动机是因为电商网站近年来获得了很大的关注。如今每个人都进入了网购世界，包括我们自己。因此，我们认为分析此类网站的因果效应会很有趣，并实际了解某些变量如何影响客户的流失率。该项目的主要目标是找到导致客户流失的因素。

### 数据集

- 这里使用的数据集是公开可用的[Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)数据集
- 包含5630行和20列
- 目标变量是**ChurnFlag**，如果客户已流失，值为1，否则为0
- 处理变量：SatisfactionFlag（满意度标志）、ComplainFlag（投诉标志）

*SatisfactionScore = 1 时 SatisfactionFlag = 1；SatisfactionScore = [2,5] 时 SatisfactionFlag = 0

### 假设

- 满意度标志有两个值：真或假。如果客户不满意，他们更可能流失
- 投诉是不满意的表现，可能导致流失
- 距离上次订单天数越高表示客户长期未使用电商网站/应用（订单数量较低），因此更可能流失
- 上月订单数量较低表示电商网站/应用使用较少，因此相比去年订单增长百分比较低
- 相比去年订单增长百分比的提高表明客户活跃，这意味着流失率会很低
- 任期是客户与组织关系的长度。任期越长表明在网站上的活动越多，因此订单数量越高，相比去年订单增长百分比越高
- 城市层级：一线城市可能有更高的订单数量，这可能降低流失的机会

### 视频链接

我们演示的视频链接：https://youtu.be/C6TPtmsYvL4