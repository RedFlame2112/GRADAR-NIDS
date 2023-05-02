from modules import *
import os
GRADAR_FILE = './GRADAR_keras.h5'
GRADAR = keras.models.load_model(GRADAR_FILE)
device = tf.test.gpu_device_name()
if device != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device))
def run_inference_on_pcap(pcap_file_path, device):
    """
    Example code to Run inference on a pcap file
    :param pcap_file_path: path to the pcap file
    :param output_dir: path to the output directory
    :return: None
    Edit this as need be; Main idea is to bring the resulting tensor from extraction into the shape
    (num_samples, 1, 77)

    """
    print('Loading model...')
    model = keras.load_model(GRADAR_FILE)
    print('Model loaded')
    print('Running inference on pcap file and extracting features...')
    #### convert pcap into csv by utilizing the command '$ cicflowmeter -f pcap_file_path -c flows.csv

    #Run command
    cmd = 'cicflowmeter -f'+ pcap_file_path +'-c inference/flows.csv'
    os.system(cmd)
    df = pd.read_csv('example.csv')
    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    ##### Used for the specific example from example.csv
    # df = df.drop(' Label', axis=1)
    # df = df.drop('Destination Port', axis=1)
    X = df.to_numpy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], 1, X.shape[1]).astype(np.float32)
    with tf.device(device):
        y_pred = GRADAR.predict(X)
        y_pred = np.rint(y_pred)

    unique_values_pred, counts_pred = np.unique(y_pred, return_counts=True)
    max_count_index_pred = np.argmax(counts_pred)
    majority_value_pred = unique_values_pred[max_count_index_pred]
    
    return majority_value_pred

def main():
   pcap_file_path = './example.pcap' ###PCAP file Not included for example
   majority_value_pred = run_inference_on_pcap(pcap_file_path, device)
   if(majority_value_pred):
      print("This sample is malicious")
   else:
      print("This sample is benign")