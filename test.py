import unittest
import pandas as pd
from ml import MLSystem

class TestMLSystem(unittest.TestCase):
    def setUp(self):
        self.ml_system = MLSystem()
        self.ml_system.load_data('train.csv', 'test.csv')

    def test_training(self):
       
        try:
            self.ml_system.train()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"El entrenamiento falló: {e}")

    def test_evaluate(self):
       
        self.ml_system.train()
        accuracy = self.ml_system.evaluate()
        self.assertGreater(accuracy, 0.5)  # Debe ser mayor que 50%

    def test_predict(self):
        
        self.ml_system.train()
        
        # Cargar las características del archivo test.csv
        test_data = pd.read_csv('test.csv')
        X_test = test_data.iloc[:, :-1].values  # Suponiendo que las características están en todas las columnas menos la última
        
        # Realizar predicciones usando las características de test.csv
        predictions = self.ml_system.predict(X_test)

        # Verificar que se realizaron las predicciones
        self.assertEqual(len(predictions), len(X_test)) 

if __name__ == '__main__':
    # Ejecuta las pruebas
    unittest.main()
