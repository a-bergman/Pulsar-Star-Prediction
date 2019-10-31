# Problem Statement

-----

A pulsar is a type of stellar remnant formed from the collapsed core of a giant star. They have very strong magnetic fields and eject pulses of electromagnetic radiation out like a lighthouse; we can only detect the jet when it is angled at the Earth.  Pulsars are very import to modern physics and astrophysics: they have been used to study nuclear physics, General Relativity, and have even been instrumental in the proof of gravitational waves.  Because of the vastness of space and how distant they are, pulsars are uncommon and very difficult to identify.  Have an automated process to identify them using machine learning methods can the speed at which these stars are discovered and can help advance our scientific knowledge.

-----

# Executive Summary

-----

We found the data on [Kaggle](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star) and was was taken from the Hight Time Resolution Universe Survey.

As a group, we all are interested in astronomy and especially the exotic objects, such as black holes.  When we saw this data, we knew we wanted to try to predict pulsars.

The data was remarkably clean: there were no missing values and no irregularities we were able to detect.  However, the description of the data does mention that it was checked manually, which could be a source of human error.  The only real cleaning we performed was to shorten column names.  When we were visualizing the data, we noticed a few points of interest: some features had a near perfect normal distribution and there were some very strong correlations between the features; we kept both in mind when it came to pre-processing.  Finally, we noticed that the data is _extremely_ imbalanced: our target class only makes up 9.1% of our data.

When we were feature engineering, we considered what we saw in our data visualizations.  Firstly, we decided to square the features that had a near perfect normal distribution which made the features more normal.  Secondly, we created interaction features based off the features with the strongest correlations.  Because of the way we had created features, we knew we had to create subsets of the data with the new features.  We modeled on three sub-sets: original features, squared features, and original features with interaction columns.

We decided to use only one type of model: a feed forward neural network.  We chose to use this model because they are very good at dealing with very imbalanced data.  Because neural networks can become extremely overfit, we took steps to make sure that our model would not become overfit: we incorporated L<sub>2</sub> regularization and kept the hidden layers to a minimum.  Because we have three sub-sets of the data, a neural network will be run on each sub-set and the best model will be whichever set has the highest metric scores.  Once we have determined the best model, we will generate a ROC curve and plot the metric scores.


-----

# Conclusions & Recommendations

-----

After running the models on each sub-set, we were able to determine our best model as the neural network with the interaction features.  The scores were very similar, but we placed a lot of emphasis on the ROC-AUC score because it indicates the degree of overlap between the two classes: our best model had a score of 0.91193, which indicates our classes are very distinct.  The other metric scores were high and there was negligible overfitting as well.

We were able to predict pulsars accurately and with minimal false negatives and have confidence in our model's performance.  That being said, we do believe that there is still room for improvement.  We would like to experiment with different regularization techniques, such as dropout, and either down-sampling the majority class or up-sampling the minority class.  We feel that further experimentation with the interaction columns will improve performance as well.

-----

# Links

The Google Slides presentation can be found [here](https://docs.google.com/presentation/d/18mKm0ecHR6tM23GHu8oQDrhBvg1xl2E08E2PzdNKfBk/edit?usp=sharing).

The Medium article can be found [here](https://medium.com/@andrew.j.bergman/predicting-pulsar-stars-996e22440cf7).