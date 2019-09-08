# Political Leanings in the 2016 Presidential Election - Predicting Party Preferences using a Tuned Generalized Additive Model

Project Participants: Ramtin Talebi, Richa Chaturvedi, Kartik Papatla, Mirai Shah

### Introduction

This project, part of a Kaggle competition and the final project for a course in generalized linear models, predicts whether an individual voter would support Democrats in the 2016 election using a generalized additive model.

Obtained from BlueLabs, the original dataset has 47 different variables, ranging from education level to whether or not the person in question plays golf. Submissions were ultimately judged based on log-loss predictive error on the holdout set.

### Approach

In the weeks and months leading up to the 2016 election, a number of pundits and political writers wrote
off the possibility of a Trump victory due to the Trump campaign's underdeveloped voter analytics team.
Following the election, however, the narrative flipped – suddenly experts agreed that it was the
Democrats who had spent too much time looking at data, and not enough time talking to voters. The truth
actually lies somewhere in between. Analytics can inform, not replace, political campaigns on both the
national and the local level. To gain a better sense

Thus, to gain a better sense of political sentiment, our goal in this project is to predict whether an
individual voter would support Democrats in the 2016 election. Obtained from BlueLabs, the original
dataset has 47 different variables, ranging from education level to whether or not the person in question
plays golf. Apart from the education predictor, in which 10% of its values were missing, the dataset
contains mostly complete observations. Only 5 out of these 47 variables are continuous and quantitative
(ppi, median_census_income, cnty_pct_evangelical, cnty_pcy_religious), while the other 42 are
categorical or binary. 

Ultimately, after performing EDA, we constructed a GAM (General Additive Model) with carefully selected main effects, tuned parameters, and significant interaction terms for our final model, yielding a log-loss score of 0.58096.
