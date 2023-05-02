# GRADAR-NIDS

GRADAR (short for GPU-backed Radar) is a Network Intrusion Detection System (NIDS) that can analyze packets that have been sniffed on a network (via a packet sniffing tool such as Wireshark), and have features extracted from the open source biflow generator Cicflowmeter-v3. Afterwards, the GRADAR can run an inference on the sniffed sample and detect whether or not there may have been significant malicious activity recorded on the given network. Since IoT devices are getting more and more mainstream, there is a call to enact certain measures of security with regards to these massive networks of IoT devices, such as your home router or even your smart fridge. IoT devices run on compute-limited resources and hence, it can be quite hard to implement on IoT devices. The model we have proposed is an architecture that is able to extract data from temporal relationships between the sample, and also map/inference other features. We implemented a timeseries-inspired approach using 2 LSTM (Long short-term memory) layers that act as a time series analyzer for sequential packets, such as those that have appeared in DoS/DDoS attacks and Denial of Sleep attacks on IoT devices.

For our dataset, in order to take into account multiple modern scenarios of network attacks on large IoT and cloud networks, we have utilized the CSE-CIC-IDS2018 dataset. Combined across samples of packets measured from February 14 over to March 2nd 2018, this dataset has approximately 16 million network samples each with ground-truth labels. We were able to over/undersample underrepresented attack samples and shortened our dataset so that we can maintain a high industry standard of >90% accuracy on both test AND train data. Furthermore, we want to have a reasonably sized dataset to train on so that our inferences are not prone to overfitting, and so we tried to quantize/prune the model effectively and reduce the number of epochs that we have to train it. Overall, our final results were a 98% accuracy on both test and train data, and fair generalization to other cloud/IoT network data that we can extract features from using the [CiCFlowMeter](https://github.com/ahlashkari/CICFlowMeter) open source tool.



For next steps, we plan to implement optimized inferencing and generalized model prediction using other frameworks closer to the metal over python such as Rust/C(++). Stay tuned for more :)
