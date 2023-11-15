from model_training.LSTM.LSTM_forecast_precipitation import *
import os

def write_results_to_file(path, neurons_list, rmse_test, predictions):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'/dados.txt', 'w') as arquivo:
        # Escrever os números e listas no arquivo
        arquivo.write(f'{rmse_test}\n')
        arquivo.write(' '.join(map(str, neurons_list)) + '\n')
        arquivo.write(' '.join(map(str, predictions)) + '\n')
    
    return

neurons_big_list = [
    [(32,0),(16,0),(8,0)],
    [(32,0.15),(16,0.15),(8,0.15)],
    [(32,0.3),(16,0.3),(8,0.3)],
    [(64,0),(16,0),(8,0)],
    [(64,0.15),(16,0.15),(8,0.15)],
    [(64,0.3),(16,0.3),(8,0.3)],
    [(64,0),(32,0),(8,0)],
    [(64,0.15),(32,0.15),(8,0.15)],
    [(64,0.3),(32,0.3),(8,0.3)],
    [(64,0),(32,0),(16,0)],
    [(64,0.15),(32,0.15),(16,0.15)],
    [(64,0.3),(32,0.3),(16,0.3)],
    [(64,0),(32,0),(16,0),(8,0)],
    [(64,0.15),(32,0.15),(16,0.15),(8,0.15)],
    [(64,0.3),(32,0.3),(16,0.3),(8,0.3)],
    [(64,0), (32,0), (32,0), (8,0)],
    [(64,0.15), (32,0.15), (32,0.15), (8,0.15)],
    [(64,0.3), (32,0.3), (32,0.3), (8,0.3)],
    [(64,0), (32,0), (32,0),(16,0), (8,0)],
    [(64,0.15), (32,0.15), (32,0.15),(16,0.15), (8,0.15)],
    [(64,0.3), (32,0.3), (32,0.3),(16,0.3), (8,0.3)],
    [(128,0),(64,0),(32,0),(16,0),(8,0)],
    [(128,0.15),(64,0.15),(32,0.15),(16,0.15),(8,0.15)],
    [(128,0.3),(64,0.3),(32,0.3),(16,0.3),(8,0.3)],
    [(64,0),(32,0),(32,0),(32,0),(16,0),(16,0)],
    [(64,0.15),(32,0.15),(32,0.15),(32,0.15),(16,0.15),(16,0.15)],
    [(64,0.3),(32,0.3),(32,0.3),(32,0.3),(16,0.3),(16,0.3)],
    [(64,0),(32,0),(32,0),(32,0),(32,0),(16,0),(16,0),(8,0)],
    [(64,0.15),(32,0.15),(32,0.15),(32,0.15),(32,0.15),(16,0.15),(16,0.15),(8,0.15)],
    [(64,0.3),(32,0.3),(32,0.3),(32,0.3),(32,0.3),(16,0.3),(16,0.3),(8,0.3)]
]

file_path = r'meteorologia\diarios_rio\A601.csv'
output_folder_path = r'hyper_tunning\LSTM_results'
windows_size = 48
df = reading_and_cleaning(file_path)
X_train, y_train, imputed_test, test_set_scaled, training_set_scaled, scaler = train_test_split(df, windows_size)
counter = 0

for neurons_list in neurons_big_list:
    counter += 1
    model = define_model(X_train, neurons_list)
    model.fit(X_train, y_train, epochs = 300, batch_size = 16)
    predictions = model_forecast(model, training_set_scaled, test_set_scaled, scaler, windows_size)
    rmse_test = np.sqrt(mean_squared_error(imputed_test, predictions))
    
    plt.figure() 
    plt.plot(imputed_test.index, imputed_test, color = 'red', label = 'series')
    plt.plot(imputed_test.index, predictions, color = 'blue', label = 'predicted values')
    plt.title('LSTM - forecast')
    plt.xlabel('Time')
    plt.ylabel('Solar Irradiance')
    plt.legend()

    path = output_folder_path + '/' + str(counter)
    write_results_to_file(path, neurons_list, rmse_test, predictions)
    plt.savefig(path+'/grafico.png')
