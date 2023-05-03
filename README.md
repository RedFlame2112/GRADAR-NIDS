# GRADAR-NIDS


## Abstract
(Track II: Accelerated Learning Module on GPU)
In the field of IOT (Internet of Things), computing systems and devices are being deployed on the edge of the internet by the day. As a result, the demand for the security of such large systems is becoming a higher priority and this paper will serve to produce an efficient model that can be easily deployed to edge devices and provide an additional layer of security that's so strongly required in this growing industry.

Our goal in particular is to solve a threat classification problem in regards to large computer/edge-device networks. A Network Intrusion Detection System (NIDS) must be able to perform an industry-level accurate ($\geq96 \%$ success rate) evaluation of a transmission's threat level, and be compact enough to be deployable to any number of edge devices, which have constrained computation limits. To achieve this accuracy, I have compiled an LSTM-CNN based deep-learning model that takes in a sequence of actions on a given network, and outputs a binary classifier that evaluates whether or not this sequence contributes to any malicious activity that might possibly harm the performance of the network, or extract private data from it. We are performing post-training integer quantization and GPU-optimization in CoLab in regards to the training process in order to be able to deploy this model to edge devices so that it can run within the constrained resources of edge devices such as Raspberry PIs, and so that the model can still run on outdated devices without significant ($\rho \leq 0.005$) loss of accuracy.

GRADAR (short for GPU-backed Radar) is a Network Intrusion Detection System (NIDS) that can analyze packets that have been sniffed on a network (via a packet sniffing tool such as Wireshark), and have features extracted from the open source biflow generator Cicflowmeter-v3. Afterwards, the GRADAR can run an inference on the sniffed sample and detect whether or not there may have been significant malicious activity recorded on the given network. Since IoT devices are getting more and more mainstream, there is a call to enact certain measures of security with regards to these massive networks of IoT devices, such as your home router or even your smart fridge. IoT devices run on compute-limited resources and hence, it can be quite hard to implement on IoT devices. The model we have proposed is an architecture that is able to extract data from temporal relationships between the sample, and also map/inference other features. We implemented a timeseries-inspired approach using 2 LSTM (Long short-term memory) layers that act as a time series analyzer for sequential packets, such as those that have appeared in DoS/DDoS attacks and Denial of Sleep attacks on IoT devices. We also implemented annotated quantized layers for the CNN portion of the network in order to reduce training and inference time, as well as better memory management in regards to the model's h5 size.

For our dataset, in order to take into account multiple modern scenarios of network attacks on large IoT and cloud networks, we have utilized the CSE-CIC-IDS2018 dataset. Combined across samples of packets measured from February 14 over to March 2nd 2018, this dataset has approximately 16 million network samples each with ground-truth labels. We were able to over/undersample underrepresented attack samples and shortened our dataset so that we can maintain a high industry standard of >90% accuracy on both test AND train data. Furthermore, we want to have a reasonably sized dataset to train on so that our inferences are not prone to overfitting, and so we tried to quantize/prune the model effectively and reduce the number of epochs that we have to train it. Overall, our final results were a 98% accuracy on both test and train data, and fair generalization to other cloud/IoT network data that we can extract features from using the [CiCFlowMeter](https://github.com/datthinh1801/cicflowmeter/tree/mainCICFlowMeter) open source tool.

One of our biggest challenges with the data was the fact that it was extremely large, and had a strong overall imbalance with regards to the actual attack vs benign classes. For each class, our data had an imbalance of almost over 80% of the data being oversampled in favor of benign packets. In practice, most models working with this dataset were therefore biased to classify most packets as benign, leading to an extremely high false negative value in the confusion matrix. In order to prevent that, we ran 2 main algorithms in our preprocessing stage. The first algorithm was SMOTE, or the Synthetic Minority Oversampling TEchnique. The way SMOTE works is by "first select(ing) a minority class instance a at random and finds its k nearest minority class neighbors. The synthetic instance is then created by choosing one of the k nearest neighbors b at random and connecting a and b to form a line segment in the feature space. The synthetic instances are generated as a convex combination of the two chosen instances a and b" - [1](https://amzn.to/32K9K6d). Next, we also utilized undersampling on the very large benign classified packets so that the number can match with the amount of malicious classified packets. This approach combining both over and undersampling was much more effective over purely undersampling the data, resulting in a model trained to not overfit and generalize well to other data. We have analyzed a test sample taken from 2015 that simulated a cloud network DDoS, and despite the slightly different features that were extracted at the time, our model still managed to accurately classify the sample to a satisfactory degree.

The interesting idea behind our model is that unlike standard models that utilize approahces such as ANN/Autoencoders like [Kitsune](https://github.com/ymirsky/Kitsune-py) and other approaches, outlined below:
![Image](https://cdn.discordapp.com/attachments/819417070185480202/1103160766338699274/image.png)

we have implemented a supervised learning approach utilizing feature extraction tools such as CiCFlowmeter to make our model supervised and hence reliable for labelling data extracted in this format, utilizing our LSTM-CNN style model in order to accurately label sequential samples of any size that can be fed into our network in the form of pcap samples.

Overall, we have employed the following metrics to measure our model's success.

Post train/testing, we have the following 4 variables we can extract from our classification confusion matrix:
- $TP$ : the number of true positives classified (packet predicted malicious was malicious)
- $TN$ : the number of true negatives classified (packet predicted benign, was benign)
- $FN$ : the number of false negatives classified (packet predicted benign, was malicious)
- $FP$ : the number of false positives classified (packet predicted malicious, was benign)

Our overall accuracy was calculated as $$a = \frac{TP+TN}{TP+TN+FN+FP}$$ and based on the confusion matrix, this was found to be $0.9792$, which was better than what we intended. Presicion was calculated as $$p = \frac{TP}{TP+FP} = 0.9811$$ recall was $$r = \frac{TP}{TP+FN} = 0.9773$$. Finally, we calculated our classiication F1 score, which was $$2 \cdot \frac{pr}{p+r} = 0.9792$$ which shows how well our model performed with classifying both benign and malicious samples even though our data was originally extremely imbalanced.

For next steps, we plan to implement optimized inferencing and generalized model prediction using other frameworks closer to the metal over python such as Rust/C(++). We also plan on furthering  Stay tuned for more :)


# References: 
Link 1: Pg 47 of "Imbalanced Learning: Foundations, Algorithms, and Applications", 2013

[Mirsky, Doitshman, Elovici, Shabtai: "Kitsune: An Ensemble of Autoencoders for Online
Network Intrusion Detection"](https://arxiv.org/pdf/1802.09089.pdf)

[Smys, Basar, Wang: "Hybrid Intrusion Detection System for Internet of Things (IoT)"](https://irojournals.com/iroismac/V2/I4/02.pdf)

[Saurabh, Sood, Kumar, Singh, R. Vyas, O.P. Vyas, Khondoker : "LBDMIDS: LSTM Based Deep Learning Model for
Intrusion Detection Systems for IoT Networks"](https://arxiv.org/pdf/2207.00424.pdf)

[CSE-CIC-IDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)
