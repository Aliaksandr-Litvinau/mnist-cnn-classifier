from data.mnist_data import load_data
from models.model import create_model
from models.train_model import train_model
from evaluation.evaluate_model import evaluate_model


def main():
    # Loading data
    x_train, y_train, x_test, y_test = load_data()

    # Model creation
    model = create_model()

    # Model training
    train_model(model, x_train, y_train)

    # Model Performance Evaluation
    accuracy, loss = evaluate_model(model, x_test, y_test)

    # Output of results
    print(f'Test accuracy: {accuracy}')
    print(f'Test loss: {loss}')


if __name__ == "__main__":
    main()
